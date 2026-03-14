import QuantLib as ql
import numpy as np
from typing import Optional, List

from fixedincomelib.analytics.bond_utilities import BondUtils
from fixedincomelib.date import Date, Period, TermOrTerminationDate
from fixedincomelib.date.utilities import accrued
from fixedincomelib.market.basics import AccrualBasis
from fixedincomelib.market.data_conventions import CompoundingMethod
from fixedincomelib.market.registries import FundingIdentifier
from fixedincomelib.product.linear_products import (
    ProductBond,
    ProductFxForward,
    ProductOvernightIndexBasisSwap,
    ProductZeroSpread,
)
from fixedincomelib.product.utilities import LongOrShort, PayOrReceive
from fixedincomelib.valuation import *
from fixedincomelib.product import (
    ProductBulletCashflow,
    ProductRFRFuture,
    ProductOvernightIndexCashflow,
    ProductFixedAccrued,
    InterestRateStream,
    ProductRFRSwap,
)
from fixedincomelib.valuation.valuation_engine import ValuationRequest
from fixedincomelib.valuation.valuation_parameters import *
from fixedincomelib.yield_curve.yield_curve_model import YieldCurve
from fixedincomelib.yield_curve.valuation_engine_analytics import (
    ValuationEngineAnalyticsOvernightIndex,
)


class ValuationEngineProductBulletCashflow(ValuationEngineProduct):
    def __init__(
        self,
        model: YieldCurve,
        valuation_parameters_collection: ValuationParametersCollection,
        product: ProductBulletCashflow,
        request: ValuationRequest,
    ):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.currency_ = product.currency
        self.termination_date_ = product.termination_date
        self.sign_ = 1.0 if product.long_or_short == LongOrShort.LONG else -1.0
        self.notional_ = product.notional
        self.vpc_: ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_: FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)

    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):
        self.df_ = 1.0
        if self.value_date <= self.termination_date_:
            scaler = self.sign_ * self.notional_
            if self.value_date == self.termination_date_:
                self.value_ = self.cash_ = scaler
            else:
                self.df_ = self.model_.discount_factor(self.funding_index_, self.termination_date_)
                self.value_ = scaler * self.df_

    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        if self.value_date < self.termination_date_:
            self.model_.discount_factor_gradient_wrt_state(
                self.funding_index_,
                self.termination_date_,
                local_grad,
                scaler * self.sign_ * self.notional_,
                True,
            )
        self.model_.resize_gradient(gradient)
        if accumulate:
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
        else:
            gradient[:] = local_grad

    def create_cash_flows_report(self) -> CashflowsReport:
        this_cf = CashflowsReport()
        this_cf.add_row(
            0,
            self.product_.product_type,
            self.val_engine_type(),
            self.notional_,
            self.sign_,
            self.termination_date_,
            self.value_ / self.df_,
            self.value_,
            self.df_,
        )
        return this_cf

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_)
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report


class ValuationEngineProductFixedAccrued(ValuationEngineProduct):
    def __init__(
        self,
        model: YieldCurve,
        valuation_parameters_collection: ValuationParametersCollection,
        product: ProductFixedAccrued,
        request: ValuationRequest,
    ):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.currency_ = product.currency
        self.effective_date_ = product.effective_date
        self.termination_date_ = product.termination_date
        self.payment_date_ = product.payment_date
        self.accrued_ = product.accrued
        self.sign_ = 1.0 if product.long_or_short == LongOrShort.LONG else -1.0
        self.notional_ = product.notional
        self.vpc_ = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_: FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)

    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):
        self.df_ = 1.0
        self.value_ = 0.0
        self.cash_ = 0.0
        if self.value_date <= self.payment_date_:
            scaler = self.sign_ * self.notional_ * self.accrued_
            if self.value_date == self.payment_date_:
                self.value_ = self.cash_ = scaler
            else:
                self.df_ = self.model_.discount_factor(self.funding_index_, self.payment_date_)
                self.value_ = scaler * self.df_

    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        if self.value_date < self.termination_date_:
            self.model_.discount_factor_gradient_wrt_state(
                self.funding_index_,
                self.payment_date_,
                local_grad,
                scaler * self.sign_ * self.notional_ * self.accrued_,
                True,
            )
        self.model_.resize_gradient(gradient)
        if accumulate:
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
        else:
            gradient[:] = local_grad

    def create_cash_flows_report(self) -> CashflowsReport:
        this_cf = CashflowsReport()
        this_cf.add_row(
            0,
            self.product_.product_type,
            self.val_engine_type(),
            self.notional_,
            self.sign_,
            self.payment_date_,
            self.value_ / self.df_,
            self.value_,
            self.df_,
        )
        return this_cf

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_)
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report


class ValuationEngineProductZeroSpread(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product: ProductZeroSpread, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.currency_ = product.currency
        self.termination_date_ = product.termination_date
        self.sign_ = 1.0 if product.long_or_short == LongOrShort.LONG else -1.0
        self.notional_ = product.notional
        self.index_ = product.index
        self.zero_spread_ = product.zero_rate
        self.time_to_expiry_ = accrued(self.value_date, self.termination_date_)
        self.vpc_: ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_: FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_: FundingIdentifier = self.funding_vp_.get_funding_index(self.currency_)

    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):
        if self.value_date <= self.termination_date_:
            self.df_funding_ = self.model_.discount_factor(self.funding_index_, self.termination_date_)
            self.df_base_ = self.model_.discount_factor(self.index_, self.termination_date_)
            self.value_ = self.sign_ * self.notional_ * (
                self.df_funding_ / self.df_base_ - np.exp(-self.zero_spread_ * self.time_to_expiry_)
            )
        else:
            raise Exception("Zero spread product has to be outright.")

    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        if self.value_date <= self.termination_date_:
            self.model_.discount_factor_gradient_wrt_state(
                self.funding_index_,
                self.termination_date_,
                local_grad,
                scaler * self.sign_ * self.notional_ / self.df_base_,
                True,
            )
            self.model_.discount_factor_gradient_wrt_state(
                self.index_,
                self.termination_date_,
                local_grad,
                -scaler * self.sign_ * self.notional_ * self.df_funding_ / self.df_base_ / self.df_base_,
                True,
            )
        self.model_.resize_gradient(gradient)
        if accumulate:
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
        else:
            gradient[:] = local_grad

    def create_cash_flows_report(self) -> CashflowsReport:
        raise Exception("ProductZeroSpread does not support cashflow report")

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_)
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report

    def grad_at_par(self):
        local_grad = []
        self.model_.resize_gradient(local_grad)
        if self.value_date <= self.termination_date_:
            df_ratio = self.df_funding_ / self.df_base_
            self.model_.discount_factor_gradient_wrt_state(
                self.funding_index_,
                self.termination_date_,
                local_grad,
                -1.0 / self.time_to_expiry_ * df_ratio / self.df_base_,
                True,
            )
            self.model_.discount_factor_gradient_wrt_state(
                self.index_,
                self.termination_date_,
                local_grad,
                1.0 / self.time_to_expiry_ * df_ratio * self.df_funding_ / self.df_base_ / self.df_base_,
                True,
            )
        return local_grad


class ValuationEngineProductRfrFuture(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product: ProductRFRFuture, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.currency_ = product.currency
        self.effective_date_ = product.effective_date
        self.termination_date_ = product.termination_date
        self.strike_ = product.strike
        self.sign_ = 1.0 if product.long_or_short == LongOrShort.LONG else -1.0
        self.notional_ = product.notional
        self.on_index_ = product.on_index
        self.vpc_: ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_: FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)
        tortd = TermOrTerminationDate(self.termination_date_.ISO())
        self.index_engine_ = ValuationEngineAnalyticsOvernightIndex(
            self.model_, self.vpc_, self.on_index_, self.effective_date_, tortd, CompoundingMethod.COMPOUND
        )
        self.forward_rate_ = 0.0

    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):
        self.dvdfwd_ = 0.0
        if self.value_date <= self.effective_date_:
            self.index_engine_.calculate_value()
            self.forward_rate_ = self.index_engine_.value()
            self.value_ = self.sign_ * self.notional_ * (100.0 - 100.0 * self.forward_rate_ - self.strike_)
            if self.value_date != self.effective_date_:
                self.dvdfwd_ = -100.0 * self.sign_ * self.notional_

    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        if self.value_date < self.effective_date_:
            self.index_engine_.calculate_risk(local_grad, scaler * self.dvdfwd_, True)
        self.model_.resize_gradient(gradient)
        if accumulate:
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
        else:
            gradient[:] = local_grad

    def create_cash_flows_report(self) -> CashflowsReport:
        this_cf = CashflowsReport()
        this_cf.add_row(
            0,
            self.product_.product_type,
            self.val_engine_type(),
            self.notional_,
            self.sign_,
            self.effective_date_,
            self.value_,
            self.value_,
            1.0,
            fixing_date=self.termination_date_,
            start_date=self.effective_date_,
            end_date=self.termination_date_,
            index_or_fixed=self.on_index_.name(),
            index_value=self.forward_rate_,
        )
        return this_cf

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_)
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report

    def par_rate_or_spread(self) -> float:
        return self.forward_rate_

    def pv01(self):
        return self.sign_ * self.notional_

    def grad_at_par(self) -> np.ndarray:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        if self.value_date < self.effective_date_:
            self.index_engine_.calculate_risk(local_grad, -100.0, True)
        return local_grad


class ValuationEngineInterestRateStream(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product: InterestRateStream, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.stream_: InterestRateStream = product
        self.currencies_ = product.currency
        self.currency_ = self.currencies_[0] if isinstance(self.currencies_, list) else self.currencies_
        self.vpc_: ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_: FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)
        self.fixed_rate_ = getattr(product, "fixed_rate_", None)
        self.float_index_ = getattr(product, "float_index_", None)
        self.index_engines_: List[Optional[ValuationEngineAnalyticsOvernightIndex]] = []
        for i in range(product.num_cashflows()):
            cf = product.cashflow(i)
            if isinstance(cf, ProductOvernightIndexCashflow):
                tortd = TermOrTerminationDate(cf.termination_date_.ISO())
                self.index_engines_.append(
                    ValuationEngineAnalyticsOvernightIndex(
                        self.model_, self.vpc_, cf.on_index, cf.effective_date, tortd, cf.compounding_method
                    )
                )
            else:
                self.index_engines_.append(None)
        self.dfs_: List[float] = [1.0] * product.num_cashflows()
        self.payoffs_: List[float] = [0.0] * product.num_cashflows()
        self.fwds_: List[Optional[float]] = [None] * product.num_cashflows()
        self.accruals_: List[Optional[float]] = [None] * product.num_cashflows()

    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def cashflow_payoff(self, cf) -> float:
        if isinstance(cf, ProductOvernightIndexCashflow):
            eng = self.index_engines_[self._cf_idx_]
            eng.calculate_value()
            fwd = eng.value()
            self.fwds_[self._cf_idx_] = fwd
            dc = cf.on_index.dayCounter()
            acc = dc.yearFraction(cf.effective_date, cf.termination_date)
            self.accruals_[self._cf_idx_] = acc
            return cf.notional * acc * (fwd + cf.spread)
        if isinstance(cf, ProductFixedAccrued):
            if self.fixed_rate_ is None:
                raise Exception("No fixed_rate in InterestRateStream")
            self.accruals_[self._cf_idx_] = cf.accrued
            return cf.notional * self.fixed_rate_ * cf.accrued
        raise Exception(f"Unsupported cashflow type: {type(cf)}")

    def calculate_value(self):
        self.value_ = 0.0
        self.cash_ = 0.0
        n = self.stream_.num_cashflows()
        for i in range(n):
            self._cf_idx_ = i
            cf = self.stream_.cashflow(i)
            pay_date = getattr(cf, "payment_date", None)
            if pay_date is None:
                pay_date = cf.last_date
            self.dfs_[i] = 1.0
            self.payoffs_[i] = 0.0
            self.fwds_[i] = None
            self.accruals_[i] = None
            if self.value_date > pay_date:
                continue
            payoff = self.cashflow_payoff(cf)
            self.payoffs_[i] = payoff
            if self.value_date == pay_date:
                self.value_ += payoff
                self.cash_ += payoff
            else:
                df = self.model_.discount_factor(self.funding_index_, pay_date)
                self.dfs_[i] = df
                self.value_ += payoff * df

    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        n = self.stream_.num_cashflows()
        for i in range(n):
            self._cf_idx_ = i
            cf = self.stream_.cashflow(i)
            pay_date = getattr(cf, "payment_date", None)
            if pay_date is None:
                pay_date = cf.last_date
            if self.value_date >= pay_date:
                continue
            payoff = self.payoffs_[i]
            df = self.dfs_[i]
            if isinstance(cf, ProductOvernightIndexCashflow):
                eng = self.index_engines_[i]
                acc = self.accruals_[i]
                if acc is None:
                    dc = cf.on_index.dayCounter()
                    acc = dc.yearFraction(cf.effective_date, cf.termination_date)
                dv_dfwd = cf.notional * acc * df
                eng.calculate_risk(local_grad, scaler * dv_dfwd, True)
            self.model_.discount_factor_gradient_wrt_state(
                self.funding_index_, pay_date, local_grad, scaler * payoff, True
            )
        self.model_.resize_gradient(gradient)
        if accumulate:
            for k in range(len(gradient)):
                gradient[k] += local_grad[k]
        else:
            gradient[:] = local_grad

    def create_cash_flows_report(self) -> CashflowsReport:
        this_cf = CashflowsReport()
        n = self.stream_.num_cashflows()
        for i in range(n):
            cf = self.stream_.cashflow(i)
            pay_date = getattr(cf, "payment_date", None)
            if pay_date is None:
                pay_date = cf.last_date
            sign = 1.0 if cf.notional >= 0 else -1.0
            notional_abs = abs(cf.notional)
            fixing_date = None
            start_date = None
            end_date = None
            index_or_fixed = None
            index_value = None
            accrued_amt = self.accruals_[i]
            if isinstance(cf, ProductOvernightIndexCashflow):
                fixing_date = cf.termination_date
                start_date = cf.effective_date
                end_date = cf.termination_date
                index_or_fixed = cf.on_index.name()
                index_value = self.fwds_[i]
            elif isinstance(cf, ProductFixedAccrued):
                start_date = cf.effective_date
                end_date = cf.termination_date
                index_or_fixed = "FIXED"
                index_value = self.fixed_rate_
            this_cf.add_row(
                0,
                self.product_.product_type,
                self.val_engine_type(),
                notional_abs,
                sign,
                pay_date,
                self.payoffs_[i],
                self.payoffs_[i] * self.dfs_[i]
                if self.value_date < pay_date
                else (self.payoffs_[i] if self.value_date == pay_date else 0.0),
                self.dfs_[i],
                fixing_date=fixing_date,
                start_date=start_date,
                end_date=end_date,
                accrued=accrued_amt,
                index_or_fixed=index_or_fixed,
                index_value=index_value,
            )
        return this_cf

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currencies_)
        if isinstance(self.currencies_, list):
            report.set_pv(self.currency_, self.value_)
            report.set_cash(self.currency_, self.cash_)
        else:
            report.set_pv(self.currencies_, self.value_)
            report.set_cash(self.currencies_, self.cash_)
        return report


class ValuationEngineProductRfrSwap(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product: ProductRFRSwap, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.currency_ = product.currency
        self.effective_date_ = product.effective_date
        self.termination_date_ = product.termination_date
        self.on_index_ = product.on_index
        self.fixed_rate_ = product.fixed_rate
        self.spread_ = product.spread
        self.pay_or_rec_ = product.pay_or_rec
        self.compounding_method = product.compounding_method
        self.fixed_leg_sign_ = -1.0 if self.pay_or_rec_ == PayOrReceive.PAY else 1.0
        self.floating_leg_sign_ = -self.fixed_leg_sign_
        self.notional_ = self.product_.notional
        self.fixed_rate_ = self.product_.fixed_rate
        self.vpc_: ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_: FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)
        self.fixed_leg_engine_ = ValuationEngineInterestRateStream(self.model_, self.vpc_, self.product_.fixed_leg, request)
        self.floating_leg_engine_ = ValuationEngineInterestRateStream(
            self.model_, self.vpc_, self.product_.floating_leg, request
        )
        self.fixed_value_ = 0.0
        self.float_value_ = 0.0
        self.fixed_cash_ = 0.0
        self.float_cash_ = 0.0
        self.annuity_ = 0.0
        self.par_rate_or_spread_ = 0.0

    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):
        self.fixed_leg_engine_.calculate_value()
        self.floating_leg_engine_.calculate_value()
        self.fixed_value_ = self.fixed_leg_engine_.value_
        self.float_value_ = self.floating_leg_engine_.value_
        self.fixed_cash_ = self.fixed_leg_engine_.cash_
        self.float_cash_ = self.floating_leg_engine_.cash_
        self.value_ = self.fixed_leg_sign_ * self.fixed_value_ + self.floating_leg_sign_ * self.float_value_
        self.cash_ = self.fixed_leg_sign_ * self.fixed_cash_ + self.floating_leg_sign_ * self.float_cash_
        self.annuity_ = self.fixed_value_ / self.notional_ / self.fixed_rate_
        self.par_rate_or_spread_ = self.float_value_ / self.notional_ / self.annuity_

    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        self.fixed_leg_engine_.calculate_first_order_risk(local_grad, scaler=self.fixed_leg_sign_ * scaler, accumulate=True)
        self.floating_leg_engine_.calculate_first_order_risk(
            local_grad, scaler=self.floating_leg_sign_ * scaler, accumulate=True
        )
        self.model_.resize_gradient(gradient)
        if accumulate:
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
        else:
            gradient[:] = local_grad

    def create_cash_flows_report(self) -> CashflowsReport:
        this_cf = CashflowsReport()
        n_fix = self.product_.fixed_leg.num_cashflows()
        for i in range(n_fix):
            cf = self.product_.fixed_leg.cashflow(i)
            pay_date = getattr(cf, "payment_date", None)
            if pay_date is None:
                pay_date = cf.last_date
            this_cf.add_row(
                0,
                self.product_.product_type,
                self.val_engine_type(),
                cf.notional,
                self.fixed_leg_sign_,
                pay_date,
                self.fixed_leg_engine_.payoffs_[i],
                self.fixed_leg_engine_.payoffs_[i] * self.fixed_leg_engine_.dfs_[i]
                if self.value_date < pay_date
                else (self.fixed_leg_engine_.payoffs_[i] if self.value_date == pay_date else 0.0),
                self.fixed_leg_engine_.dfs_[i],
                fixing_date=None,
                start_date=getattr(cf, "effective_date", None),
                end_date=getattr(cf, "termination_date", None),
                accrued=self.fixed_leg_engine_.accruals_[i],
                index_or_fixed="FIXED",
                index_value=self.fixed_rate_,
            )
        n_flt = self.product_.floating_leg.num_cashflows()
        for i in range(n_flt):
            cf = self.product_.floating_leg.cashflow(i)
            pay_date = getattr(cf, "payment_date", None)
            if pay_date is None:
                pay_date = cf.last_date
            index_or_fixed = getattr(cf, "on_index", None)
            index_or_fixed = index_or_fixed.name() if index_or_fixed is not None else None
            this_cf.add_row(
                1,
                self.product_.product_type,
                self.val_engine_type(),
                cf.notional,
                self.floating_leg_sign_,
                pay_date,
                self.floating_leg_engine_.payoffs_[i],
                self.floating_leg_engine_.payoffs_[i] * self.floating_leg_engine_.dfs_[i]
                if self.value_date < pay_date
                else (self.floating_leg_engine_.payoffs_[i] if self.value_date == pay_date else 0.0),
                self.floating_leg_engine_.dfs_[i],
                fixing_date=getattr(cf, "termination_date", None),
                start_date=getattr(cf, "effective_date", None),
                end_date=getattr(cf, "termination_date", None),
                accrued=self.floating_leg_engine_.accruals_[i],
                index_or_fixed=index_or_fixed,
                index_value=self.floating_leg_engine_.fwds_[i],
            )
        return this_cf

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_)
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report

    def par_rate_or_spread(self) -> float:
        return self.par_rate_or_spread_

    def pv01(self) -> float:
        return self.fixed_value_ / self.fixed_rate_ * self.fixed_leg_sign_

    def grad_at_par(self) -> np.ndarray:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        a = self.fixed_value_ / self.fixed_rate_
        self.floating_leg_engine_.calculate_first_order_risk(local_grad, scaler=1.0 / a, accumulate=True)
        self.fixed_leg_engine_.calculate_first_order_risk(
            local_grad, scaler=-self.float_value_ / self.fixed_rate_ / a / a, accumulate=True
        )
        return local_grad


class ValuationEngineProductOvernightIndexBasisSwap(ValuationEngineProduct):
    def __init__(
        self,
        model: YieldCurve,
        valuation_parameters_collection: ValuationParametersCollection,
        product: ProductOvernightIndexBasisSwap,
        request: ValuationRequest,
    ):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.product_: ProductOvernightIndexBasisSwap = product
        self.currency_ = product.currency
        self.effective_date_ = product.effective_date
        self.termination_date_ = product.termination_date
        self.on_index_1_ = product.on_index_1
        self.on_index_2_ = product.on_index_2
        self.spread_ = product.spread
        self.pay_or_rec_ = product.pay_or_rec
        self.notional_ = product.notional
        self.compounding_method_ = product.compounding_method
        self.float_1_leg_sign_ = 1.0 if self.pay_or_rec_ == PayOrReceive.RECEIVE else -1.0
        self.float_2_leg_sign_ = -self.float_1_leg_sign_
        self.vpc_: ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_: FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)
        self.floating_leg_1_wo_basis_engine_ = ValuationEngineInterestRateStream(
            self.model_, self.vpc_, self.product_.floating_leg_1_wo_basis, request
        )
        self.floating_leg_1_basis_engine_ = ValuationEngineInterestRateStream(
            self.model_, self.vpc_, self.product_.floating_leg_1_basis, request
        )
        self.floating_leg_2_engine_ = ValuationEngineInterestRateStream(
            self.model_, self.vpc_, self.product_.floating_leg_2, request
        )
        self.float_1_wo_basis_value_ = 0.0
        self.float_1_basis_value_ = 0.0
        self.float_1_value_ = 0.0
        self.float_2_value_ = 0.0
        self.float_1_wo_basis_cash_ = 0.0
        self.float_1_basis_cash_ = 0.0
        self.float_1_cash_ = 0.0
        self.float_2_cash_ = 0.0
        self.annuity_ = 0.0
        self.par_rate_or_spread_ = 0.0

    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):
        self.floating_leg_1_wo_basis_engine_.calculate_value()
        self.floating_leg_1_basis_engine_.calculate_value()
        self.floating_leg_2_engine_.calculate_value()
        self.float_1_wo_basis_value_ = self.floating_leg_1_wo_basis_engine_.value_
        self.float_1_basis_value_ = self.floating_leg_1_basis_engine_.value_
        self.float_1_value_ = self.float_1_wo_basis_value_ + self.float_1_basis_value_
        self.float_2_value_ = self.floating_leg_2_engine_.value_
        self.float_1_wo_basis_cash_ = self.floating_leg_1_wo_basis_engine_.cash_
        self.float_1_basis_cash_ = self.floating_leg_1_basis_engine_.cash_
        self.float_1_cash_ = self.float_1_wo_basis_cash_ + self.float_1_basis_cash_
        self.float_2_cash_ = self.floating_leg_2_engine_.cash_
        self.value_ = self.float_1_leg_sign_ * self.float_1_value_ + self.float_2_leg_sign_ * self.float_2_value_
        self.cash_ = self.float_1_leg_sign_ * self.float_1_cash_ + self.float_2_leg_sign_ * self.float_2_cash_
        self.annuity_ = self.float_1_basis_value_ / self.spread_ if self.spread_ != 0.0 else 0.0
        self.par_rate_or_spread_ = (
            (self.float_2_value_ - self.float_1_wo_basis_value_) / self.annuity_ if self.annuity_ != 0.0 else 0.0
        )

    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        self.floating_leg_1_wo_basis_engine_.calculate_first_order_risk(
            local_grad, scaler=self.float_1_leg_sign_ * scaler, accumulate=True
        )
        self.floating_leg_1_basis_engine_.calculate_first_order_risk(
            local_grad, scaler=self.float_1_leg_sign_ * scaler, accumulate=True
        )
        self.floating_leg_2_engine_.calculate_first_order_risk(
            local_grad, scaler=self.float_2_leg_sign_ * scaler, accumulate=True
        )
        self.model_.resize_gradient(gradient)
        if accumulate:
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
        else:
            gradient[:] = local_grad

    def create_cash_flows_report(self) -> CashflowsReport:
        this_cf = CashflowsReport()
        n_leg_1 = self.product_.floating_leg_1_wo_basis.num_cashflows()
        for i in range(n_leg_1):
            cf = self.product_.floating_leg_1_wo_basis.cashflow(i)
            pay_date = getattr(cf, "payment_date", None)
            if pay_date is None:
                pay_date = cf.last_date
            payoff = self.floating_leg_1_wo_basis_engine_.payoffs_[i] + self.floating_leg_1_basis_engine_.payoffs_[i]
            pv = (
                self.floating_leg_1_wo_basis_engine_.payoffs_[i] * self.floating_leg_1_wo_basis_engine_.dfs_[i]
                + self.floating_leg_1_basis_engine_.payoffs_[i] * self.floating_leg_1_basis_engine_.dfs_[i]
                if self.value_date < pay_date
                else (payoff if self.value_date == pay_date else 0.0)
            )
            df = self.floating_leg_1_wo_basis_engine_.dfs_[i]
            index_or_fixed = getattr(cf, "on_index", None)
            index_name = index_or_fixed.name() if index_or_fixed is not None else None
            if index_name is not None:
                index_name = f"{index_name} + spread"
            this_cf.add_row(
                1,
                self.product_.product_type,
                self.val_engine_type(),
                cf.notional,
                self.float_1_leg_sign_,
                pay_date,
                payoff,
                pv,
                df,
                fixing_date=getattr(cf, "termination_date", None),
                start_date=getattr(cf, "effective_date", None),
                end_date=getattr(cf, "termination_date", None),
                accrued=self.floating_leg_1_wo_basis_engine_.accruals_[i],
                index_or_fixed=index_name,
                index_value=self.floating_leg_1_wo_basis_engine_.fwds_[i],
            )
        n_leg_2 = self.product_.floating_leg_2.num_cashflows()
        for i in range(n_leg_2):
            cf = self.product_.floating_leg_2.cashflow(i)
            pay_date = getattr(cf, "payment_date", None)
            if pay_date is None:
                pay_date = cf.last_date
            index_or_fixed = getattr(cf, "on_index", None)
            index_or_fixed = index_or_fixed.name() if index_or_fixed is not None else None
            this_cf.add_row(
                2,
                self.product_.product_type,
                self.val_engine_type(),
                cf.notional,
                self.float_2_leg_sign_,
                pay_date,
                self.floating_leg_2_engine_.payoffs_[i],
                self.floating_leg_2_engine_.payoffs_[i] * self.floating_leg_2_engine_.dfs_[i]
                if self.value_date < pay_date
                else (self.floating_leg_2_engine_.payoffs_[i] if self.value_date == pay_date else 0.0),
                self.floating_leg_2_engine_.dfs_[i],
                fixing_date=getattr(cf, "termination_date", None),
                start_date=getattr(cf, "effective_date", None),
                end_date=getattr(cf, "termination_date", None),
                accrued=self.floating_leg_2_engine_.accruals_[i],
                index_or_fixed=index_or_fixed,
                index_value=self.floating_leg_2_engine_.fwds_[i],
            )
        return this_cf

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_)
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report

    def par_rate_or_spread(self) -> float:
        return self.par_rate_or_spread_

    def pv01(self) -> float:
        return self.annuity_

    def grad_at_par(self) -> np.ndarray:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        if self.annuity_ == 0.0:
            return local_grad
        numerator = self.float_2_value_ - self.float_1_wo_basis_value_
        self.floating_leg_2_engine_.calculate_first_order_risk(local_grad, scaler=1.0 / self.annuity_, accumulate=True)
        self.floating_leg_1_wo_basis_engine_.calculate_first_order_risk(
            local_grad, scaler=-1.0 / self.annuity_, accumulate=True
        )
        if self.spread_ != 0.0:
            self.floating_leg_1_basis_engine_.calculate_first_order_risk(
                local_grad,
                scaler=-numerator / (self.annuity_ * self.annuity_ * self.spread_),
                accumulate=True,
            )
        return local_grad


class ValuationEngineProductFXForward(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product: ProductFxForward, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.currency_ = product.currency
        self.termination_date_ = product.termination_date
        self.pay_or_rec_ = product.pay_or_rec
        self.notional_ = self.product_.notional
        self.sign_ = -1.0 if product.pay_or_rec_ == PayOrReceive.PAY else 1.0
        self.fx_pair_ = product.fx_pair
        self.strike_ = product.strike
        self.vpc_: ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_: FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)
        self.value_ = 0.0
        self.cash_ = 0.0
        self.par_rate_or_spread_ = 0.0

    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):
        self.fx_rate_ = self.model_.fx_rate(self.fx_pair_, self.termination_date_)
        self.value_ = self.sign_ * self.notional_ * (self.fx_rate_ - self.strike_)
        if self.value_date == self.termination_date_:
            self.cash_ = self.value_
        self.par_rate_or_spread_ = self.fx_rate_

    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        self.model_.fx_rate_gradient_wrt_state(
            self.fx_pair_,
            self.termination_date_,
            local_grad,
            scaler=scaler * self.sign_ * self.notional_,
            accumulate=True,
        )
        self.model_.resize_gradient(gradient)
        if accumulate:
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
        else:
            gradient[:] = local_grad

    def create_cash_flows_report(self) -> CashflowsReport:
        return CashflowsReport()

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_)
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report

    def par_rate_or_spread(self) -> float:
        return self.par_rate_or_spread_

    def grad_at_par(self) -> np.ndarray:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        self.model_.fx_rate_gradient_wrt_state(
            self.fx_pair_, self.termination_date_, local_grad, scaler=1.0, accumulate=True
        )
        return local_grad


class ValuationEngineProductBond(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product: ProductBond, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        self.product_: ProductBond = product
        self.face_value_ = product.face_value
        self.currencies_ = product.currency
        self.currency_ = self.currencies_[0] if isinstance(self.currencies_, list) else self.currencies_
        self.traded_price_ = self.product_.traded_price
        self.vpc_: ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_: FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)
        self.bond_funding_ = self.funding_vp_.get_underlying_funding_by_ccy(self.currency_)
        if self.bond_funding_ is None:
            self.bond_funding_ = self.funding_index_
        v = FundingIndexParameter({"Funding Index": self.bond_funding_.name_})
        self.vpc_bond_funding_ = ValuationParametersCollection([v])
        prod_ntl: ProductBulletCashflow = self.product_.principal
        self.engine_ntl = ValuationEngineProductRegistry().new_valuation_engine(
            self.model_, prod_ntl, self.vpc_bond_funding_, request
        )
        self.accrued_engines: List[Optional[ValuationEngineProductFixedAccrued]] = []
        for coupon in product.coupons_cf:
            eng = ValuationEngineProductRegistry().new_valuation_engine(
                self.model_, coupon, self.vpc_bond_funding_, request
            )
            self.accrued_engines.append(eng)
        self.dfs_: List[float] = [1.0] * product.num_cashflows()
        self.payoffs_: List[float] = [0.0] * product.num_cashflows()
        self.fwds_: List[Optional[float]] = [None] * product.num_cashflows()
        self.accruals_: List[Optional[float]] = [None] * product.num_cashflows()

    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):
        self.value_ = 0.0
        self.cash_ = 0.0
        for eng in self.accrued_engines:
            eng.calculate_value()
            self.value_ += eng.value_ * self.face_value_
            self.cash_ += eng.cash_ * self.face_value_
        self.engine_ntl.calculate_value()
        self.value_ += self.engine_ntl.value_ * self.face_value_
        self.cash_ += self.engine_ntl.cash_ * self.face_value_
        self.bf_price_ = self.value_
        self.df_settlement_ = self.model_.discount_factor(self.bond_funding_, self.product_.settlement_date)
        self.value_ /= self.df_settlement_
        self.value_ -= self.traded_price_
        self.df_settlement_csa = self.model_.discount_factor(self.funding_index_, self.product_.settlement_date)
        self.value_ *= self.df_settlement_csa

    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False) -> None:
        local_grad = []
        self.model_.resize_gradient(local_grad)
        scaled = scaler / self.df_settlement_ * self.df_settlement_csa
        for eng in self.accrued_engines:
            eng.calculate_first_order_risk(local_grad, scaler=scaled * self.face_value_, accumulate=True)
        self.engine_ntl.calculate_first_order_risk(local_grad, scaler=scaled * self.face_value_, accumulate=True)
        if self.value_date < self.product_.settlement_date:
            self.model_.discount_factor_gradient_wrt_state(
                self.bond_funding_,
                self.product_.settlement_date,
                local_grad,
                scaler * (-self.bf_price_ / self.df_settlement_**2) * self.df_settlement_csa,
                accumulate=True,
            )
            self.model_.discount_factor_gradient_wrt_state(
                self.funding_index_,
                self.product_.settlement_date,
                local_grad,
                scaler * (self.bf_price_ / self.df_settlement_ - self.traded_price_),
                accumulate=True,
            )
        self.model_.resize_gradient(gradient)
        if accumulate:
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
        else:
            gradient[:] = local_grad

    def create_cash_flows_report(self) -> CashflowsReport:
        this_cf = CashflowsReport()
        for eng in self.accrued_engines:
            this_cf.add_row(
                0,
                self.product_.product_type,
                self.val_engine_type(),
                abs(eng.notional_),
                eng.sign_,
                eng.payment_date_,
                eng.value_ / eng.df_,
                eng.value_,
                eng.df_,
                eng.effective_date_,
                eng.termination_date_,
                accrued=eng.accrued_,
                index_or_fixed="FIXED",
                index_value=abs(eng.notional_),
            )
        this_cf.add_row(
            1,
            self.product_.product_type,
            self.val_engine_type(),
            self.engine_ntl.notional_,
            self.engine_ntl.sign_,
            self.engine_ntl.termination_date_,
            self.engine_ntl.value_ / self.engine_ntl.df_,
            self.engine_ntl.value_,
            self.engine_ntl.df_,
        )
        return this_cf

    def grad_at_par(self):
        dirty_price = self.bf_price_
        _, dpdy, _ = BondUtils.price_to_yield(self.product_, dirty_price, clean=False)
        grad = []
        self.model_.resize_gradient(grad)
        scaled = 1 / self.df_settlement_
        for eng in self.accrued_engines:
            eng.calculate_first_order_risk(grad, scaler=scaled * self.face_value_, accumulate=True)
        self.engine_ntl.calculate_first_order_risk(grad, scaler=scaled * self.face_value_, accumulate=True)
        if self.value_date < self.product_.settlement_date:
            self.model_.discount_factor_gradient_wrt_state(
                self.bond_funding_,
                self.product_.settlement_date,
                grad,
                scaled * (-self.bf_price_ / self.df_settlement_**2),
                accumulate=True,
            )
        for i in range(len(grad)):
            grad[i] = grad[i] / dpdy
        return grad

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currencies_)
        if isinstance(self.currencies_, list):
            report.set_pv(self.currency_, self.value_)
            report.set_cash(self.currency_, self.cash_)
        else:
            report.set_pv(self.currencies_, self.value_)
            report.set_cash(self.currencies_, self.cash_)
        return report


### register
ValuationEngineProductRegistry().register(
    (YieldCurve._model_type.to_string(), ProductBulletCashflow._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductBulletCashflow,
)
ValuationEngineProductRegistry().register(
    (YieldCurve._model_type.to_string(), ProductFixedAccrued._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductFixedAccrued,
)
ValuationEngineProductRegistry().register(
    (YieldCurve._model_type.to_string(), ProductRFRFuture._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductRfrFuture,
)
ValuationEngineProductRegistry().register(
    (YieldCurve._model_type.to_string(), InterestRateStream._product_type, AnalyticValParam._vp_type),
    ValuationEngineInterestRateStream,
)
ValuationEngineProductRegistry().register(
    (YieldCurve._model_type.to_string(), ProductRFRSwap._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductRfrSwap,
)
ValuationEngineProductRegistry().register(
    (YieldCurve._model_type.to_string(), ProductOvernightIndexBasisSwap._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductOvernightIndexBasisSwap,
)
ValuationEngineProductRegistry().register(
    (YieldCurve._model_type.to_string(), ProductZeroSpread._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductZeroSpread,
)
ValuationEngineProductRegistry().register(
    (YieldCurve._model_type.to_string(), ProductBond._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductBond,
)
ValuationEngineProductRegistry().register(
    (YieldCurve._model_type.to_string(), ProductFxForward._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductFXForward,
)
