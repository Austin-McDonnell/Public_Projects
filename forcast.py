import pandas as pd
import numpy as np
import QuantLib as ql
from collections import namedtuple
from typing import Optional, List, Union, NamedTuple, Tuple
from dataclasses import dataclass, astuple, fields
from scipy.optimize import minimize

pd.options.display.float_format = '{:,.2f}'.format


@dataclass
class Schedule:
    period: int = 0
    pre_payment: float = 0
    charge_offs: float = 0
    payment: float = 0
    interest: float = 0
    principal: float = 0
    outstanding_balance: float = 0


@dataclass
class CashFlows:
    period: int = 0
    asset_cashflow: float = 0
    interest_payment: float = 0
    advance_maintenance_cashflow: float = 0
    operating_net_income: float = 0
    debt_payment: float = 0
    net_cashflow: float = 0
    outstanding_equity: float = 0
    outstanding_asset_balance: float = 0
    outstanding_debt: float = 0


class LoanPortfolio:
    '''
    Creates a single loan portfolio forecast
    '''
    def __init__(self,
                 total_principal: float,
                 loan_term_months: int,
                 interest_rate: float,
                 prepayment_each_month: float = 0.0,
                 losses_per_month: float = 0.0,
                 loss_toggle_period_start: Optional[int] = None,
                 ):
        self.total_principal = total_principal
        self.loan_term_months = loan_term_months
        self.interest_rate = interest_rate
        self.prepayment_each_month = prepayment_each_month
        self.losses_per_month = losses_per_month
        self.loss_toggle_period_start = loss_toggle_period_start

        # Can adjust for variable periods using ql
        self.num_payments = self.loan_term_months
        self.payment_period_rate = self.interest_rate / 12

        self.payment_period_amount: Optional[float] = None

    def get_period_payment(self) -> float:
        monthly_payment = (self.total_principal *
                           (self.payment_period_rate * (1 + self.payment_period_rate) ** self.num_payments) /
                           ((1 + self.payment_period_rate) ** self.num_payments - 1)
                           )
        return monthly_payment

    def pre_payment_amount(self, current_outstanding_balance) -> float:
        return current_outstanding_balance * self.prepayment_each_month

    def losses_amount(self, current_outstanding_balance) -> float:
        return current_outstanding_balance * self.losses_per_month

    def forcast(self, pre_payment_toggle: bool = False, loss_toggle: bool = False):
        if not self.payment_period_amount:
            self.payment_period_amount = self.get_period_payment()

        full_schedule = []
        outstanding_balance = self.total_principal
        full_schedule.append(
            Schedule(period=0, pre_payment=0, charge_offs=0, payment=0,
                     interest=0, principal=0, outstanding_balance=outstanding_balance)
        )

        for pay_period in range(1, self.num_payments + 1):
            if self.loss_toggle_period_start:
                # Can create varied loss toggle pay periods
                if self.loss_toggle_period_start <= pay_period:
                    loss_toggle = True

            period_interest_amount = self.payment_period_rate * outstanding_balance
            period_principal_amount = self.payment_period_amount - period_interest_amount

            charge_off_amount = 0

            pre_payment_amount = 0

            if pre_payment_toggle:
                pre_payment_amount = self.pre_payment_amount(outstanding_balance)

            if loss_toggle:
                charge_off_amount = self.losses_amount(outstanding_balance)

            # Add up the total outstanding balance deduction for the payment period
            payment_period_balance_impact = (period_principal_amount + pre_payment_amount + charge_off_amount)

            # If the outstanding balance amount is less than the expected payment_period_balance_impact then take
            # take the sum of the parts and calculate the remaining principal payment after pre_payment and losses
            if outstanding_balance < payment_period_balance_impact:

                payment_period_balance_impact = outstanding_balance

                if pre_payment_toggle:
                    pre_payment_amount = payment_period_balance_impact * self.prepayment_each_month
                if loss_toggle:
                    charge_off_amount = payment_period_balance_impact * self.losses_per_month

                period_principal_amount = payment_period_balance_impact - pre_payment_amount - charge_off_amount

            outstanding_balance -= payment_period_balance_impact
            schedule = Schedule(
                period=pay_period,
                pre_payment=pre_payment_amount,
                charge_offs=charge_off_amount,
                payment=self.payment_period_amount,
                interest=period_interest_amount,
                principal=period_principal_amount,
                outstanding_balance=outstanding_balance
            )
            full_schedule.append(schedule)

            # TODO: Address when the calculated outstanding balance is greater than the previous
            if outstanding_balance <= 0:
                break

        return pd.DataFrame(full_schedule).set_index('period')


class RollingLoanPortfolio(LoanPortfolio):
    '''
    Applies singluar portfolios into a layered rolling window structure
    '''

    def __init__(self, roll_period: int, **kwargs):
        super().__init__(**kwargs)
        self.roll_period = roll_period

    def cumulative_roll(self, pre_payment_toggle: bool = False, loss_toggle: bool = False):
        roll_period_cashflows_list = []
        for roll_period in list(range(self.roll_period + 1)):
            roll_period_cash_flow_df = self.forcast(pre_payment_toggle=pre_payment_toggle, loss_toggle=loss_toggle)
            roll_period_cash_flow_df.index = roll_period_cash_flow_df.index + roll_period

            roll_period_cashflows_list.append(roll_period_cash_flow_df)
        return pd.concat(roll_period_cashflows_list).groupby('period').sum()


class Structure:

    def __init__(
            self,
            advance_rate: float,
            cost_of_debt: float,
            starting_cash: float = 0,
            start_debt_pay_down_period: Optional[int] = None,
    ):
        self.advance_rate = advance_rate
        self.cost_of_debt = cost_of_debt
        self.starting_cash = starting_cash
        self.start_debt_pay_down_period = start_debt_pay_down_period

        self.outstanding_equity: Optional[float] = None
        self.outstanding_debt: Optional[float] = None
        self.equity_funding: Optional[float] = None

        self.cash_flows_list: List[CashFlows] = []

    def calculate_starting_funding_structure(self, asset_principal_balance: float, debt_charge: float):
        # This could be optimized or changed depending upon the cost of Equity compared to cost of debt
        # TODO: build equity minimization function to minimize the amount of equity need to put up

        self.outstanding_equity = (1 - self.advance_rate) * debt_charge
        self.outstanding_debt = debt_charge * self.advance_rate
        # Take into account the value of the starting equity
        self.starting_cash += self.outstanding_equity
        self.cash_flows_list.append(
            CashFlows(
                advance_maintenance_cashflow=self.starting_cash,
                operating_net_income=self.starting_cash,
                net_cashflow=self.starting_cash,
                outstanding_equity=self.starting_cash,
                outstanding_asset_balance=asset_principal_balance,
                outstanding_debt=self.outstanding_debt)
        )

    def calculate_period_interest_payment(self):
        # TODO: This is a simplified interest payment method on the debt
        return self.cost_of_debt / 12 * self.outstanding_debt

    def calculate_period_asset_cashflow(self, row: NamedTuple):
        return row.pre_payment + row.payment - row.charge_offs

    def calculate_advance_rate_maintenance(self, row: float):
        '''
        Calculates the advance rate maintenance amount if the outstanding debt % of Assets + Equity and then forces an
        equity injection to make up the difference

        **Could take the equity cure and make this a Debt Cure if past the debt payment period**
        **Could create a Minimizer for minimizing the Debt or Equity depending upon the cost of
        Equity injections vs the cost of the debt**
        '''
        #
        equity_cure = 0
        if self.outstanding_debt / (row.outstanding_balance + self.outstanding_equity) > self.advance_rate:
            print(f'Advance Rate Maintenance for period: {row.period}')

            equity_cure = (self.outstanding_debt - row.outstanding_balance * self.advance_rate) / self.advance_rate

        return equity_cure

    def pay_down_outstanding_debt(self, row):
        '''
        The debt is paid down by using as much equity as possible without going negative

        Could Solve the Advance Rate maintenance fee by paying down debt instead if the period has been passed; this again
        could optimize the profits dependent upon both costs
        '''
        debt_payment = 0
        if self.start_debt_pay_down_period:
            if self.start_debt_pay_down_period <= row.period:
                if self.outstanding_equity > 0 and self.outstanding_debt > 0:
                    if self.outstanding_debt > self.outstanding_equity:
                        debt_payment = self.outstanding_equity
                    else:
                        debt_payment = self.outstanding_debt
        return debt_payment

    def forecast_structure_cashflows(self, asset_cashflows_df: pd.DataFrame, debt_schedule: pd.Series):
        '''
        Runs through the asset cashflows, and debt schedule given to calculate the overall structure cashflows for each
        period
        '''
        cash_flows = pd.concat([asset_cashflows_df, debt_schedule], axis=1).fillna(0)
        cash_flows.index.name = 'period'

        self.calculate_starting_funding_structure(asset_principal_balance=cash_flows.iloc[0].outstanding_balance,
                                                  debt_charge=cash_flows.iloc[0].debt_charge)

        for row in cash_flows.iloc[1:, :].reset_index().itertuples(index=False):
            # Calculate the interest amount before adding on the current period debt amount
            interest_payment = self.calculate_period_interest_payment()

            self.outstanding_debt += row.debt_charge
            equity_cure = self.calculate_advance_rate_maintenance(row)
            period_asset_cashflow = self.calculate_period_asset_cashflow(row)
            net_interest_cashflow = period_asset_cashflow - interest_payment
            debt_payment = self.pay_down_outstanding_debt(row)
            net_cashflow = net_interest_cashflow + equity_cure

            self.outstanding_debt -= debt_payment
            self.outstanding_equity += (net_cashflow - debt_payment)

            self.cash_flows_list.append(
                CashFlows(
                    period=row.period,
                    asset_cashflow=period_asset_cashflow,
                    interest_payment=interest_payment,
                    advance_maintenance_cashflow=equity_cure,
                    operating_net_income=net_interest_cashflow,
                    debt_payment=debt_payment,
                    net_cashflow=net_cashflow,
                    outstanding_equity=self.outstanding_equity,
                    outstanding_asset_balance=row.outstanding_balance,
                    outstanding_debt=self.outstanding_debt)
            )
        df = pd.DataFrame(self.cash_flows_list).set_index('period')

        return df


if __name__ == '__main__':
    test = LoanPortfolio(
        total_principal=1000000,
        loan_term_months=36,
        interest_rate=0.2,
        prepayment_each_month=0.02,
        losses_per_month=0.01,
        loss_toggle_period_start=None
    )

    results = test.forcast(
        pre_payment_toggle=False,
    )
    print(results[['pre_payment', 'charge_offs', 'payment', 'interest', 'principal']].sum())

    rolling_test = RollingLoanPortfolio(
        roll_period=12,
        total_principal=1000000,
        loan_term_months=36,
        interest_rate=0.2,
        prepayment_each_month=0.02,
        losses_per_month=0.01,
        loss_toggle_period_start=None
    )

    cumulative_results = rolling_test.cumulative_roll()
    print(cumulative_results)

    structure = Structure(
        advance_rate=0.75,
        cost_of_debt=0.08,
        start_debt_pay_down_period=12
    )

    debt_schedule = pd.Series(np.array([1000000] * 12 + [0] * 37, dtype=float), index=np.arange(0, 49), name='debt_charge')
    structure_cashflows = structure.forecast_structure_cashflows(cumulative_results, debt_schedule)
    print(structure_cashflows)

    print('Done')
