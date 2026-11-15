# Volatility Convention Conversion in Pure / SQL

## Problem: No Unbounded 1D Solvers in SQL

In classic C++ or Python pricing libraries, converting between volatility conventions (for example, Black-76 lognormal vol $\sigma_{\text{LN}}$ to Bachelier normal vol $\sigma_N$, or vice versa) is typically implemented by:
1. Evaluating the option price under the source convention.
2. Running an iterative 1D root-finder (Brent, bisection, or variable-iteration Newton) against the target pricing formula to recover the implied volatility.

Unbounded `while` loops do not compile cleanly into single-pass SQL `SELECT` projections.

## Pattern: Asymptotic Expansion + Fixed-Step Polish

In Pure/SQL, replace the 1D loop with a **closed-form asymptotic initial guess** followed by **1–2 fixed Newton (or Halley) polish steps** using the true pricing and vega functions:

1. **Asymptotic Initial Guess ($\sim 10^{-4}$–$10^{-5}$ relative accuracy)**
   - **Black $\sigma_{\text{LN}} \to$ Normal $\sigma_N$** (for $F, K > 0$, or shifted $F+s, K+s > 0$):
     $$\sigma_N^{(0)} = \sigma_{\text{LN}} \frac{F - K}{\ln(F / K)} \left(1 - \frac{\sigma_{\text{LN}}^2 T}{24}\right)$$
     (with the ATM Taylor limit $\sigma_{\text{LN}} F (1 - \sigma_{\text{LN}}^2 T / 24)$ as $F \to K$).
   - **Normal $\sigma_N \to$ Black $\sigma_{\text{LN}}$**:
     $$\sigma_{\text{LN}}^{(0)} = \sigma_N \frac{\ln(F / K)}{F - K} \left(1 + \frac{\sigma_N^2 T}{24 F K}\right)$$

2. **Fixed-Step Polish Using True Functions**
   - Unroll 1 or 2 Newton steps via pure `let` bindings:
     $$\sigma^{(k+1)} = \sigma^{(k)} - \frac{P_{\text{target}}(\sigma^{(k)}) - P_{\text{source}}}{{\text{Vega}}_{\text{target}}(\sigma^{(k)})}$$
   - Because Newton's method is quadratically convergent (and Halley's method is cubically convergent), starting from a $\sim 10^{-4}$ guess gives $\sim 10^{-8}$ after 1 step and machine precision ($\sim 10^{-15}$) after 2 steps—all as a branch-free inline SQL expression.

## Automatic Differentiation (`:diff`) Considerations

- **Direct differentiation**: `:diff` can differentiate directly through a 1-step polish expression, though differentiating through multiple unrolled Newton steps increases SQL expression size.
- **Implicit Function Theorem (preferred for AD)**: When differentiating a converged implied vol $\sigma_N$ with respect to market inputs $x$ (e.g., forward rate $F$ or $\sigma_{\text{LN}}$), differentiate the zero-residual identity $P_{\text{Bach}}(\sigma_N) - P_{\text{Black}}(\sigma_{\text{LN}}) = 0$ once:
  $$\frac{\partial \sigma_N}{\partial x} = \frac{\partial P_{\text{Black}} / \partial x - \partial P_{\text{Bach}} / \partial x}{\text{Vega}_{\text{Bach}}(\sigma_N)}$$
  This avoids backpropagating through the polish steps in SQL while remaining exact at the root.

## Relationship to `projects/pricing/model.pure`

- For **SABR**, `pricing::sabrNormalVol` computes the implied normal volatility directly from $(\alpha, \beta, \rho, \nu)$ via Hagan's expansion, so no Black-to-Normal inversion is required during pricing—and $\beta = 0$ handles negative rates where lognormal Black is undefined.
- The expansion + fixed-step polish pattern is intended for **market quote ingestion or convention switching** when external quotes are supplied in Black or shifted-Black vol and must be converted to Normal vol (or vice versa).
