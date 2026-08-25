# Theory

**hrHSA** is built around one central idea: habitat selection is not inferred from where an animal was observed alone, but from how observed use differs from a biologically defined set of alternatives. The analytical framework changes as the definition of those alternatives becomes more local and movement constrained.

The package therefore supports three closely related views of the same ecological process:

| Question | Availability | Likelihood | hrHSA route |
| --- | --- | --- | --- |
| Which parts of a broader landscape are used disproportionately? | A spatial domain such as an individual's range or study area | Use--availability logistic model / point-process approximation | RSF |
| Which reachable endpoint was chosen at the next movement step? | Alternatives generated from the current location and movement kernel | Conditional logistic / categorical softmax | SSF |
| How do habitat and environmental conditions modify movement itself? | Movement-generated alternatives with proposal correction | Integrated conditional-choice model | iSSF |

Frequentist and Bayesian inference are available across these routes to different degrees. They are not different ecological theories: they are alternative ways to estimate the same underlying selection and movement relationships, with the Bayesian formulations additionally representing hierarchical variation and posterior uncertainty explicitly.

## 1. Ecological foundations of habitat selection

### Habitat selection is a process, not a map category

Classical habitat-selection theory starts from a simple ecological observation: animals do not merely occur in environments; through movement, settlement, foraging, territoriality and avoidance they continually redistribute themselves among alternatives. The resulting spatial pattern is therefore an outcome of behavioural decisions interacting with environmental opportunity, competition and constraint (Fretwell & Lucas, 1969; Rosenzweig, 1981; Johnson, 1980).

This makes *habitat* organism specific. A land-cover class or raster cell is not intrinsically good, bad, selected or avoided. Its ecological meaning depends upon the resources, conditions, risks and competitors experienced by the focal organism, and may change with season, life-history stage or behavioural state. Habitat-selection analysis consequently asks about relationships between organisms and environmental conditions rather than assigning fixed values to landscape categories.

A telemetry relocation is evidence of **use**. It is not, by itself, evidence of selection, preference or habitat quality. Selection is disproportionate use relative to what was available. In the strict sense, preference refers to what would be chosen if alternatives were equally available, whereas observational telemetry studies usually estimate selection under unequal and constrained availability (Johnson, 1980; Manly et al., 2002; Lele et al., 2013).

For habitat classes $h=1,\ldots,H$, let $u_h$ be the proportion of observed use and $a_h$ the proportion available. A simple selection ratio is

$$
r_h = \frac{u_h}{a_h}.
$$

Values greater than one indicate use in excess of availability; values below one indicate use below availability. A common habitat may therefore contain many animal locations while nevertheless being used less than expected from its abundance, while a rare habitat may contain few locations but be strongly selected.

### The hierarchy of selection

Johnson's (1980) classic formulation makes the scale dependence of selection explicit. He distinguished four nested **orders of selection**:

1. **First-order selection:** the geographical range of a species.
2. **Second-order selection:** placement of an individual or social group's home range within that geographical range.
3. **Third-order selection:** use of habitat components within the home range.
4. **Fourth-order selection:** selection of particular resources or sites during specific activities such as feeding, resting or nesting.

The same environmental feature can be selected at one order and avoided at another. A mountain range may be selected when establishing a home range, for example, while the steepest terrain within that range is avoided during routine movement. Apparent contradictions among habitat-selection studies often arise because they define availability at different orders or spatial grains rather than because animals behave inconsistently.

The four orders should therefore be understood as an ecological hierarchy rather than a rigid statistical classification. Modern telemetry often resolves decisions finer than those envisioned in 1980, including individual movement steps lasting minutes or hours. The central insight nevertheless remains unchanged: **selection can only be interpreted with respect to the set of alternatives appropriate to the decision scale**.

This is why availability is part of the ecological model. It represents a hypothesis about what the animal could have used, given its position, movement capacity, temporal interval, social constraints and the spatial scale of the question. A study-area polygon, an individual range and a one-hour movement kernel answer different biological questions even if the same environmental predictors are used.

### Habitat choice, density and fitness

Habitat quality need not be independent of the animals already occupying it. Classical **Ideal Free Distribution** theory formalized habitat choice as a density-dependent process (Fretwell & Lucas, 1969). Let $N_h$ denote the number or density of animals in habitat $h$, and let $\Phi_h(N_h)$ denote the expected per-capita fitness payoff obtained there. Under competition,

$$
\frac{\partial \Phi_h(N_h)}{\partial N_h} < 0,
$$

so increasing density reduces the payoff available to each individual.

Under the ideal-free assumptions, animals can assess habitat profitability and move freely among alternatives. At equilibrium, every occupied habitat yields the same expected payoff $\lambda$:

$$
\Phi_h(N_h^*) = \lambda
\qquad
\text{for all occupied habitats }h,
$$

while an unoccupied habitat offers no greater payoff. A highly productive habitat can therefore contain more animals than a poorer habitat without providing greater realised fitness to each occupant: its higher density is precisely what reduces its payoff toward the common equilibrium.

The **Ideal Despotic Distribution** relaxes the assumption of free access. Dominant individuals may monopolise high-quality territories or exclude competitors, forcing subordinates into habitats with lower expected payoffs. Observed space use can consequently reflect habitat productivity, density, social rank, territoriality and access simultaneously. Selection patterns should not automatically be read as unconstrained expressions of preference.

This classical theory also explains why **use, density and habitat quality are not interchangeable**. Van Horne (1983) famously showed why high animal density can be a misleading indicator of habitat quality: crowding, social exclusion or demographic source--sink structure can produce high occupancy without high individual performance. Conversely, a sparsely used habitat need not be intrinsically poor if access is restricted or if it is rare. Habitat-selection coefficients quantify behavioural redistribution relative to availability; they do not by themselves establish survival, reproduction or fitness consequences.

Rosenzweig (1981) further developed habitat selection as an ecological process linking habitat profitability, competition and population distribution. In this view, the spatial distribution observed by a telemetry study is not simply a response to environmental heterogeneity. It is also part of the ecological context within which subsequent choices are made.

hrHSA does not automatically model density dependence, dominance or fitness. These processes must enter explicitly through predictors, grouping structure, interactions, demographic outcomes or a model designed for that question. The important principle is that a selection coefficient is conditional on the ecological and social context represented in the fitted data.

### Movement as constrained choice

Classical optimal-foraging theory likewise emphasized that animals choose among alternatives subject to travel costs and changing returns. MacArthur & Pianka (1966) and Charnov's (1976) marginal value theorem showed that the value of a patch cannot be separated from the cost of reaching alternatives or from the rate at which returns change through time. Modern step-selection methods do not require animals to be optimal foragers, but they inherit the same ecological lesson: **what is reachable affects what can be chosen**.

At broad scales, accessibility may be approximated by a home range or other spatial domain. At fine temporal scales, however, the animal's present location, elapsed time and movement capacity sharply constrain the next set of alternatives. The progression from RSF to SSF and iSSF is therefore not merely a progression in statistical sophistication. It represents increasingly explicit models of the ecological choice set.

Telemetry observes this continuous movement process only at discrete times. Let $S_i(t)$ denote the true location of individual $i$ and let the recorded relocations be

$$
\mathcal{D}_i = \{\tilde{s}_{ij},t_{ij}\}_{j=1}^{n_i}.
$$

The observed fixes are shaped by movement, residence time, return behaviour, fix schedule, positional error and potentially habitat-dependent fix success. Habitat-selection models should therefore be interpreted as models of relative use or local choice conditional on the observation and availability definitions supplied to them.

## 2. Resource Selection Functions

### Point-process interpretation

For a spatial domain $\Omega$, an RSF can be viewed through an inhomogeneous point-process intensity

$$
\lambda(s) = c\,f^A(s)\,w\{x(s)\},
$$

where $f^A(s)$ describes the available spatial distribution, $x(s)$ is the environmental vector at location $s$, $c$ controls total intensity and

$$
w(x) = \exp(x\beta)
$$

is the resource-selection function.

Conditional on the number of observed relocations, the implied use distribution is

$$
f^u(s)
=
\frac{f^A(s)w\{x(s)\}}
{\int_\Omega f^A(r)w\{x(r)\}\,dr}.
$$

The selection function therefore **reweights availability**. Its absolute scale is arbitrary; relative differences are the primary inferential quantity.

### Use--availability logistic approximation

In practice the spatial integral is approximated by sampling available locations and combining them with observed locations:

$$
Y_j =
\begin{cases}
1, & \text{used location},\\
0, & \text{sampled available location}.
\end{cases}
$$

A logistic model is then fitted,

$$
\operatorname{logit}\{P(Y_j=1\mid x_j)\}
=
\alpha+x_j\beta.
$$

The available points are not biological absences. Their number is controlled by the analyst and approximates the available environmental distribution. Consequently, the logistic intercept depends on the used:available sampling ratio and is not normally interpreted as an ecological occupancy probability. As the available sample becomes sufficiently dense, the slopes approximate those of the underlying point-process formulation (Johnson et al., 2006; Warton & Shepherd, 2010; Aarts et al., 2012).

### Relative selection strength

For two environmental conditions $x_1$ and $x_0$,

$$
\operatorname{RSS}(x_1,x_0)
=
\frac{w(x_1)}{w(x_0)}
=
\exp\{(x_1-x_0)\beta\}.
$$

For a one-unit difference in a single linear predictor, holding all other terms constant,

$$
\operatorname{RSS}=\exp(\beta_j).
$$

This interpretation becomes conditional when the model contains quadratic terms, categorical effects or interactions. In those models, prediction and explicit contrasts are safer than reading coefficients one at a time.

An RSF surface

$$
\hat{w}\{x(s)\}=\exp\{x(s)\hat\beta\}
$$

is likewise a map of **relative selection**, not a calibrated probability of occupancy or habitat quality. It may be normalized within a specified available domain, but the resulting distribution remains conditional on that domain.

## 3. Individuals, hierarchy and partial pooling

Telemetry studies often contain thousands or millions of fixes but far fewer independently sampled animals. Treating every relocation as a population replicate therefore produces false precision.

Individual-specific responses can be represented as

$$
\beta_i = \mu_\beta + b_i,
\qquad
b_i \sim \mathcal{N}(0,\Sigma_\beta).
$$

The population mean $\mu_\beta$ describes the average response, while $\Sigma_\beta$ describes between-individual heterogeneity and correlation among responses. Covariates describing sex, age, population or behavioural state can be added at the individual level when the ecological question requires them.

Hierarchical Bayesian models implement **partial pooling**. Animals with informative data retain strongly individual estimates, whereas poorly informed individuals are pulled toward the population distribution. This avoids both complete pooling and fitting every animal in isolation.

The distinction between a population-average response, individual-specific coefficients and between-individual heterogeneity is important. In hrHSA these quantities are exposed separately in Bayesian RSF, SSF and iSSF results rather than collapsed into one coefficient table.

Because the exponential selection function is nonlinear,

$$
E_i[\exp(x\beta_i)]
\neq
\exp\{xE_i(\beta_i)\},
$$

so the prediction for an average coefficient vector is not necessarily the same as the average prediction across heterogeneous individuals.

Ecologically, between-individual heterogeneity is not merely statistical noise. Persistent differences can reflect age, sex, reproductive state, experience, dominance, behavioural specialization or access to different environments. They may also change the effective availability experienced by different animals. A population-level coefficient should therefore be interpreted as a distributional summary of individuals, not as a literal description of an interchangeable "average animal".

## 4. Dependence and predictive validation

Successive relocations from the same animal are serially dependent. That dependence is not merely a nuisance: slow movement, residence and repeated return may be biologically meaningful. Arbitrarily thinning a trajectory until points appear independent can remove part of the process being studied.

Validation should instead respect the intended prediction task.

For RSFs, hrHSA distinguishes especially between:

- **transfer to a new individual**, evaluated with leave-one-individual-out validation;
- **finite held-out sample uncertainty**, evaluated with temporally blocked bootstrap resampling; and
- **temporal non-stationarity**, evaluated by scoring real contiguous time periods.

The continuous Boyce index is used as a rank-based RSF validation metric. If $P_k$ is the proportion of held-out used locations in prediction interval $k$ and $E_k$ is the corresponding proportion of available locations,

$$
R_k = \frac{P_k}{E_k},
$$

and the Boyce index is the Spearman correlation between prediction rank and $R_k$. A high value indicates that higher predicted selection corresponds to disproportionately greater held-out use. The full predicted-to-expected curve should be retained because a single correlation coefficient can hide where a model fails.

Frequentist and Bayesian validation intervals do not represent exactly the same uncertainty. A frequentist blocked bootstrap in hrHSA holds the fitted model fixed and resamples held-out temporal blocks, whereas the Bayesian version can additionally pair bootstrap replicates with posterior coefficient draws.

## 5. Step Selection Functions

### Availability becomes local

At fine temporal resolution, a broad RSF availability domain becomes difficult to justify. A location may lie inside an animal's range but be unreachable during the interval between two telemetry fixes. This is the step-selection analogue of Johnson's scale argument: the ecological choice set has moved from a broad landscape to the alternatives reachable from the animal's current state.

A step connects consecutive relocations,

$$
s_t \longrightarrow s_{t+1},
$$

with length

$$
L_t = \lVert s_{t+1}-s_t\rVert
$$

and turning angle $\theta_t$ relative to the preceding movement direction.

In an SSF, each observed endpoint is compared with alternatives generated from the same starting location using a movement-informed proposal. The resulting choice set is

$$
\mathcal{C}_t
=
\{s_{t+1}^{(0)},s_{t+1}^{(1)},\ldots,s_{t+1}^{(K)}\},
$$

where candidate $0$ is the observed endpoint.

For predictor vector $x_{tj}$,

$$
P(j\mid\mathcal{C}_t)
=
\frac{\exp(x_{tj}\beta)}
{\sum_k\exp(x_{tk}\beta)}.
$$

The likelihood is conditional on the choice set. A stratum intercept is therefore unnecessary: adding the same constant to every alternative in a stratum cancels under the softmax.

### Static and dynamic environmental conditions

Candidate endpoints may be annotated with static terrain variables and with time-varying fields such as temperature, uplift or wind. The ecological timing must remain explicit. In the current iSSF workflow the natural convention is:

- endpoint conditions are sampled at `end_time`;
- departure conditions are sampled at `start_time`.

Vector fields require additional care. A wind vector at an endpoint can be projected onto each candidate's geodesic bearing. Positive support corresponds to a tailwind component, negative support to headwind. When the same start-of-step wind vector is projected onto different candidate bearings, the resulting directional support still varies among alternatives and is therefore identifiable in a conditional-choice likelihood.

### Selection opportunity

A coefficient can be weakly estimated because an animal rarely encountered contrasting alternatives, even when its biological response was strong. For predictor $k$, the conditional information contributed by stratum $s$ is related to

$$
I_{s,k}=\operatorname{Var}_{p_s}(x_{s,j,k}).
$$

Low within-choice-set contrast limits information. hrHSA therefore separates **selection opportunity** from coefficient magnitude and also exposes conditional information inflation diagnostics for overlapping predictors. This distinction is particularly useful when comparing individuals that moved through different environmental landscapes.

Ecologically, selection opportunity is the local counterpart of availability. An animal cannot demonstrate a choice between conditions that it did not encounter as meaningful alternatives. Differences in estimated selection among individuals can consequently arise from differences in response, differences in opportunity, or both.

### SSF validation

For SSFs, prediction is evaluated on the conditional choice scale. hrHSA uses the gain over a uniform choice model,

$$
G_s
=
\log p_{\text{model}}(y_s)
-
\log(1/J_s),
$$

where $J_s$ is the number of alternatives in stratum $s$. Positive gain indicates better prediction than uniform choice. Exact temporal-block and leave-one-individual-out validation answer different questions from PSIS-LOO on already fitted strata and should not be treated as interchangeable.

## 6. Integrated Step Selection Functions

A conventional SSF uses a fitted movement distribution to generate available steps and then estimates habitat selection conditional on that proposal. But the observed movement distribution has itself already been shaped by habitat. iSSF addresses this by estimating movement and habitat terms jointly while correcting for the candidate-generation distribution.

This returns to the classical ecological idea that movement costs and habitat rewards are inseparable components of choice. The iSSF does not assume optimal behaviour in the sense of MacArthur & Pianka (1966) or Charnov (1976), but it makes the movement constraints that define the choice set part of the fitted ecological process rather than treating them only as sampling machinery.

### Proposal correction

Let $q_{sj}$ be the density under the movement proposal used to generate candidate $j$ in stratum $s$. hrHSA uses the proposal-corrected utility

$$
\eta_{sj}
=
x_{sj}^{T}\beta
-
\log q_{sj}.
$$

The offset

$$
o_{sj}=-\log q_{sj}
$$

is an importance-sampling correction, not an ecological coefficient. It may be centered within a stratum because any common additive constant cancels from the conditional likelihood.

This distinction is fundamental: **availability generation and the ecological movement model are not the same object**. The proposal is a computational device for drawing candidate steps; the fitted movement terms describe the ecological kernel after proposal correction.

### Movement basis

The default movement basis in hrHSA is

$$
L,\qquad \log L,\qquad \cos\theta.
$$

When the coefficients on $L$ and $\log L$ imply a proper Gamma-like step-length kernel,

$$
k = 1+\gamma_{\log L},
\qquad
\lambda=-\gamma_L,
$$

with expected displacement

$$
E[L]=\frac{k}{\lambda}.
$$

The corresponding turning term governs directional persistence. These movement quantities describe displacement between fixes; they are not the animal's total travelled distance along an unresolved path.

### Environmental movement modifiers

Conditions at departure can alter the movement kernel. For example, heat, ruggedness or wind support can interact with $L$, $\log L$ or $\cos\theta$.

A start condition that is constant for every alternative in a stratum cannot be estimated as a standalone main effect because that constant cancels from the conditional likelihood. It becomes identifiable through its interaction with candidate-varying movement terms. By contrast, a directional quantity such as start-wind support may vary among candidate bearings and can therefore have both a main effect and movement interactions.

This leads to an important interpretive distinction. Suppose ruggedness modifies both $L$ and $\log L$. The fitted linear predictor may be additive on the coefficient scale, but expected displacement is a nonlinear transformation of the movement coefficients. Consequently, movement-response curves evaluated under different environmental conditions may diverge even when the underlying interaction structure is simple. hrHSA therefore provides derived movement-response and step-length-distribution plots rather than encouraging interpretation from isolated interaction coefficients.

### Hierarchical iSSF

For individual $i$ and predictor $p$, the Bayesian iSSF uses partially pooled coefficients of the form

$$
\beta_{ip}
=
\mu_p+\sigma_pz_{ip},
\qquad
z_{ip}\sim\mathcal{N}(0,1).
$$

This allows both habitat-selection terms and movement responses to vary among animals while retaining a population distribution.

### Current validation boundary

The iSSF workflow deliberately does **not** yet expose the SSF cross-validation schemes. Correct validation must refit environmental scaling within every training fold and preserve proposal correction in held-out choice sets. Until that fold-specific preparation is implemented, hrHSA raises rather than silently leaking information from held-out strata.

## 7. Bayesian inference is an inferential layer

For observed data $\mathcal{D}$, latent quantities $\mathcal{Z}$ and parameters $\Theta$,

$$
p(\mathcal{Z},\Theta\mid\mathcal{D})
\propto
p(\mathcal{D}\mid\mathcal{Z},\Theta)
\,p(\mathcal{Z}\mid\Theta)
\,p(\Theta).
$$

The practical value of the Bayesian formulations in hrHSA is not complexity for its own sake. They support:

- partial pooling across individuals;
- posterior distributions for population means and heterogeneity;
- propagation of coefficient uncertainty into predictions and validation;
- regularization of large correlated predictor sets when a shrinkage prior is explicitly requested; and
- comparison of ecological effects on a common posterior scale.

A regularized horseshoe prior can stabilize broader candidate models, but shrinkage is not causal variable selection and its local scales are not posterior inclusion probabilities. Highly correlated predictors may support a combined ecological signal more strongly than either coefficient separately.

Dynamic environmental covariates and time-varying coefficients should also be distinguished. hrHSA currently supports time-varying environmental fields in SSF/iSSF annotation and diagnostics. A truly dynamic coefficient model,

$$
\beta_t \sim \mathcal{N}(\beta_{t-1},Q),
$$

would represent a changing selection relationship itself. Such state-space coefficient evolution is a natural theoretical extension but is not implied merely because a predictor varies through time.

## 8. What can be predicted?

The interpretation of prediction depends on the model class.

**RSF:** predicts relative selection across a defined spatial domain. Maps rank areas by fitted selection, conditional on availability and model specification.

**SSF:** predicts the relative probability of choosing one candidate endpoint over alternatives from the same start. The probabilities are local to the choice set.

**iSSF:** additionally estimates how environmental conditions change the movement kernel. Derived expected displacement and turning distributions are often more interpretable than raw movement-interaction coefficients.

None of these quantities is automatically a measure of habitat quality or fitness. Establishing that a selected environment improves survival, reproduction or population growth requires demographic information or an explicit fitness model. This separation between **selection** and **quality** is one of the most important lessons of the classical habitat literature.

Long-term utilisation distributions, residence times, crossing probabilities and home-range geometry are emergent properties of repeated movement decisions. In principle they are obtained by simulating the fitted transition process. Habitat-biased forward simulation is not yet implemented as a production hrHSA workflow, so the current package should not be described as providing these long-term simulations directly.

## 9. Choosing an analytical route

A practical decision sequence is:

1. **Define the biological decision scale.** Which order of selection or movement decision is the study intended to represent?
2. **Define availability before fitting.** The available distribution determines what selection means.
3. **Use RSF** when a broader spatial availability domain is defensible and the target is relative space use.
4. **Use SSF** when local reachability between fixes is central but the movement proposal can be treated as the availability-generating mechanism.
5. **Use iSSF** when the movement kernel itself is part of the ecological question or environmental conditions are hypothesized to modify displacement or turning.
6. **Use hierarchical Bayesian inference** when population-level inference and individual heterogeneity are both targets, or when posterior uncertainty must propagate through derived predictions.
7. **Validate according to the intended transfer task.** New time periods, new animals and new strata are different prediction problems.
8. **Interpret derived quantities on their natural scale.** Relative selection, local choice probability, expected displacement, density and habitat quality are not interchangeable.

The specialized documentation develops each route in detail:

- [Object-oriented RSF workflows](rsf_objects.md)
- [Hierarchical Bayesian RSF workflow](bayesian_rsf.md)
- [Step-selection workflows](ssf.md)
- [Integrated step-selection workflows](issf.md)
- [Dynamic environmental condition plots](ssf_dynamic_conditions.md)

## Principal references

Aarts, G., Fieberg, J. & Matthiopoulos, J. (2012). Comparative interpretation of count, presence-absence and point methods for species distribution models. *Methods in Ecology and Evolution*, **3**, 177--187.

Avgar, T., Potts, J. R., Lewis, M. A. & Boyce, M. S. (2016). Integrated step selection analysis: bridging the gap between resource selection and animal movement. *Methods in Ecology and Evolution*, **7**, 619--630.

Charnov, E. L. (1976). Optimal foraging, the marginal value theorem. *Theoretical Population Biology*, **9**, 129--136.

Fretwell, S. D. & Lucas, H. L. (1969). On territorial behavior and other factors influencing habitat distribution in birds. *Acta Biotheoretica*, **19**, 16--36.

Johnson, C. J., Nielsen, S. E., Merrill, E. H., McDonald, T. L. & Boyce, M. S. (2006). Resource selection functions based on use-availability data: theoretical motivation and evaluation methods. *Journal of Wildlife Management*, **70**, 347--357.

Johnson, D. H. (1980). The comparison of usage and availability measurements for evaluating resource preference. *Ecology*, **61**, 65--71.

Lele, S. R., Merrill, E. H., Keim, J. & Boyce, M. S. (2013). Selection, use, choice and occupancy: clarifying concepts in resource selection studies. *Journal of Animal Ecology*, **82**, 1183--1191.

MacArthur, R. H. & Pianka, E. R. (1966). On optimal use of a patchy environment. *American Naturalist*, **100**, 603--609.

Manly, B. F. J., McDonald, L. L., Thomas, D. L., McDonald, T. L. & Erickson, W. P. (2002). *Resource Selection by Animals: Statistical Design and Analysis for Field Studies*. 2nd ed. Kluwer Academic Publishers.

Morris, D. W. (2003). Toward an ecological synthesis: a case for habitat selection. *Oecologia*, **136**, 1--13.

Muff, S., Signer, J. & Fieberg, J. (2020). Accounting for individual-specific variation in habitat-selection studies: efficient estimation of mixed-effects models using Bayesian or frequentist computation. *Journal of Animal Ecology*, **89**, 80--92.

Northrup, J. M. et al. (2022). Conceptual and methodological advances in habitat-selection modeling: guidelines for ecology and evolution. *Ecological Applications*, **32**, e02470.

Rosenzweig, M. L. (1981). A theory of habitat selection. *Ecology*, **62**, 327--335.

Van Horne, B. (1983). Density as a misleading indicator of habitat quality. *Journal of Wildlife Management*, **47**, 893--901.

Warton, D. I. & Shepherd, L. C. (2010). Poisson point process models solve the "pseudo-absence problem" for presence-only data in ecology. *Annals of Applied Statistics*, **4**, 1383--1402.
