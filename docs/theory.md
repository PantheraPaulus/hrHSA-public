# Theory

## 1. Foundations

### Habitat
Habitat provides the ecological link between individual movement and population redistribution. By remaining, departing, returning or continuing onwards, animals allocate their time among heterogeneous environments. Repeated over time, these decisions produce patterns of local space use, home-range formation, migration and seasonal range occupation; aggregated across individuals, they generate population-level patterns of distribution and density. Habitat selection therefore connects the relocation of the individual with the organisation of the population (Fretwell & Lucas, 1969; Morris, 2003; Nathan et al., 2008; Mueller & Fagan, 2008; Matthiopoulos et al., 2015).

Within Habitat Selection Analysis, *habitat* is most usefully understood as a point in environmental space defined by the conjunction of conditions, resources and risks relevant to the focal organism (Morris, 2003; Matthiopoulos et al., 2020; Northrup et al., 2022). Conditions influence physiological or behavioural functioning; resources are required for maintenance, growth or reproduction; and risks reduce expected survival or reproductive success. Habitat is therefore organism-specific: the same geographic location may present different opportunities and constraints to individuals differing in state, experience or perceptual capacity.

Several related terms must be distinguished. *Use* is the realised allocation of an animal’s time or activity to a habitat. *Occupancy* denotes physical presence within a spatial unit during a defined period. *Availability* describes the habitats that could have been encountered or used, given the spatial, temporal and behavioural constraints acting upon the animal. *Selection* is differential use relative to that availability, whereas *preference*, in its strict sense, describes selection where all alternatives are equally available (Johnson, 1980; Manly et al., 2002; Lele et al., 2013).

For habitat types $h=1,\ldots,H$, let $a_h$ denote the proportion of available habitat belonging to type $h$, and let $u_h$ denote its proportion of observed use. For $a_h>0$, the elementary selection ratio is

$$
r_h = \frac{u_h}{a_h}.
$$

Use is proportional to availability when $r_h=1$, greater than expected when $r_h>1$, and less than expected when $r_h<1$. Selection is thus a relationship between use and availability, not an intrinsic property of the habitat. A widespread habitat may contain many recorded locations while receiving less use than expected from its abundance; a rare habitat may contain few observations while nevertheless receiving strongly disproportionate use.

### The Ideal Free Distribution
Although selection is enacted by individuals, habitat value depends partly upon the population already occupying it. Competitors may deplete resources, interfere with foraging, defend territories or exclude subordinate animals. Population distribution is therefore both a consequence of earlier habitat-selection decisions and part of the ecological context governing subsequent ones. Let $N_h$ denote the number or density of animals occupying habitat $h$, and let $\Phi_h(N_h)$ denote the expected per-capita fitness payoff available there. Under competition,

$$
\frac{\partial \Phi_h(N_h)}{\partial N_h} < 0,
$$

so that habitat value declines as density increases. Habitat selection is therefore inherently density dependent (Fretwell & Lucas, 1969; Rosenzweig, 1981; Morris, 2003; McLoughlin et al., 2010).

The classical formulation of this relationship is the **Ideal Free Distribution** of Fretwell and Lucas (1969). The term *ideal* assumes that individuals can distinguish the relative payoffs of available habitats; the term *free* assumes that they may move among them without prohibitive costs or exclusion. Consider a population of total size $N$ distributed among $H$ habitats, with $N_h^{*}$ individuals occupying habitat $h$ at equilibrium:

$$
\sum_{h=1}^{H} N_h^{*} = N.
$$

An ideal-free equilibrium is reached when no individual could improve its expected fitness by moving elsewhere. All occupied habitats therefore provide the same equilibrium payoff $\lambda$:

$$
\Phi_h\left(N_h^{*}\right)
=
\lambda
\qquad
\text{for all } h \text{ with } N_h^{*}>0,
$$

while any unoccupied habitat offers no greater return:

$$
\Phi_h(0)
\leq
\lambda
\qquad
\text{when } N_h^{*}=0.
$$

A habitat with a high initial payoff may consequently be occupied first, but the arrival of additional individuals reduces its per-capita value. Once its payoff falls to that available elsewhere, individuals distribute themselves among habitats in the proportions required to maintain equal expected fitness. Habitats of greater productive capacity may therefore support greater animal densities without providing greater realised fitness at equilibrium: their greater density is precisely what reduces the individual payoff to that available in alternative habitats.

For two habitats, combinations of densities satisfying

$$
\Phi_1(N_1)=\Phi_2(N_2)
$$

form an **isodar** (Morris, 1987, 1988, 2003). The isodar intercept represents differences in habitat profitability at low density, whereas its slope represents differences in the rate at which fitness declines with crowding. Ideal-free and isodar theory thus provide a formal bridge between individual habitat choice, density-dependent competition and population distribution.

### The Ideal Despotic Distribution

The assumptions of the Ideal Free Distribution provide a theoretical reference condition rather than a universal description of nature. Individuals differ in information, state and competitive ability; movement may be costly; and dominant animals may monopolise profitable habitats. Such processes may produce an **Ideal Despotic Distribution**, in which subordinate individuals occupy less profitable habitats despite the existence of superior alternatives (Fretwell & Lucas, 1969). Observed distributions may therefore reflect habitat profitability, competition, restricted access and social organisation in combination.

Habitat selection is also hierarchical and scale dependent. Johnson (1980) distinguished selection of a species’ geographical range, placement of an individual home range within that range, use of habitat components within the home range and selection of particular resources within an activity site. A habitat may be selected at one order and avoided at another, or selected for one activity and avoided during another. Availability is therefore an ecological hypothesis concerning the alternatives accessible to an animal at a particular time and scale, rather than a purely geometric property of the study area.

Habitat Selection Analysis begins from these premises. Animals respond to environmental heterogeneity through movement and space use; their decisions produce population distributions; and the resulting densities alter the value of habitats subsequently encountered. The inferential question that follows is how this relationship may be recovered from observations of animal relocations and measured environmental features.

## 2. Making it real: Towards Inference from Empirical Data

### From Relocations to Selection

Habitat selection is a behavioural process, but telemetry records only a finite sample of its spatial consequences. Let $S_i(t)$ denote the location of individual $i$ at time $t$. A tracking device observes this continuous trajectory only at discrete times $t_{ij}$, producing the relocation series

$$
\mathcal{D}_i
=
\left\{
\tilde{s}_{ij},t_{ij}
\right\}_{j=1}^{n_i},
$$

where $\tilde{s}_{ij}$ may differ from the true location because of positional error. The observed sample is further conditioned upon the fix schedule and the probability of successfully recording a location.

A concentration of relocations in one part of the landscape may therefore arise because the habitat is selected, because it is readily accessible, because the animal moves slowly within it, or because it repeatedly returns to the same area. Conversely, few relocations may indicate avoidance, restricted access, rapid passage or reduced fix success. Selection cannot be inferred from observations of use alone; the observed distribution must be compared with an explicit model of what was available to the animal.

### Resource Selection as a Point Process

Consider a spatial domain $\Omega\subset\mathbb{R}^{2}$ containing observed relocations

$$
\mathcal{S}
=
\left\{
s_1,\ldots,s_n
\right\}.
$$

Spatial use may be represented by an **inhomogeneous Poisson point process**. If $N(B)$ denotes the number of relocations observed within a region $B\subseteq\Omega$, then

$$
N(B)
\sim
\operatorname{Poisson}
\left(
\int_B \lambda(s)\,ds
\right),
$$

where $\lambda(s)>0$ is the spatial intensity of relocations.

Environmental heterogeneity is commonly introduced through a log-linear intensity:

$$
\lambda(s)
=
\exp\left[
\beta_0+x(s)\beta
\right],
$$

where $x(s)$ is a row vector of environmental covariates and $\beta$ is the corresponding vector of coefficients. This may be written as

$$
\lambda(s)
=
\exp(\beta_0)w\left(x(s)\right),
$$

with the selection function

$$
w(x)
=
\exp(x\beta).
$$

The function $w(x)$ describes how habitat alters the relative intensity of use. It is not the absolute probability that a spatial unit will be occupied or used.

Where locations differ in their accessibility, the intensity may be written more generally as

$$
\lambda(s)
=
c\,f^{A}(s)w\left(x(s)\right),
$$

where $f^{A}(s)$ is the available distribution and $c>0$ controls the expected number of observations. Conditional upon observing $n$ relocations, the corresponding use distribution is

$$
f^{u}(s)
=
\frac{
f^{A}(s)w\left(x(s)\right)
}{
\displaystyle
\int_{\Omega}
f^{A}(r)w\left(x(r)\right)\,dr
}.
$$

The available environment is thus reweighted by the selection function to produce the expected distribution of use.

The intercept has little direct biological meaning in a conventional telemetry analysis because the total number of relocations depends upon the fix schedule, study duration and number of tracked animals. The coefficients in $\beta$, by contrast, describe relative differences in the intensity of use, conditional upon the defined available distribution.

### Approximation by Logistic Regression

The likelihood of an inhomogeneous Poisson point process contains an integral over the full spatial domain:

$$
L(\beta)
\propto
\exp
\left[
-\int_{\Omega}\lambda(s)\,ds
\right]
\prod_{i=1}^{n}
\lambda(s_i).
$$

For realistic landscapes, this integral must usually be approximated numerically. In a use--availability design, environmental conditions are evaluated at the observed relocations and at locations sampled from the available distribution.

Let

$$
Y_j
=
\begin{cases}
1, & \text{if location }j\text{ is used},\\
0, & \text{if location }j\text{ is sampled from availability}.
\end{cases}
$$

The combined sample may be fitted using logistic regression:

$$
\operatorname{logit}
\left[
\Pr(Y_j=1\mid x_j)
\right]
=
\alpha+x_j\beta.
$$

Available locations are not biological absences. They represent the environmental distribution from which the animal could have selected and provide a numerical approximation to the point-process integral. As the available sample becomes sufficiently large, the logistic slopes approach those of the underlying point-process model (Johnson et al., 2006; Warton & Shepherd, 2010; Aarts et al., 2012).

The logistic intercept $\alpha$ depends upon the analyst-defined ratio of used to available locations and should not ordinarily be interpreted. Predictions should therefore be based upon the exponential selection function

$$
\hat{w}(x)
=
\exp(x\hat{\beta}),
$$

rather than the fitted logistic probability

$$
\frac{
\exp(\alpha+x\hat{\beta})
}{
1+\exp(\alpha+x\hat{\beta})
}.
$$

The number of available locations should be increased until the estimated slopes stabilise. Additional available points improve numerical integration but do not create additional biological replication.

### Interpretation of an RSF

For two environmental conditions $x_1$ and $x_0$, their relative selection strength is

$$
\operatorname{RSS}(x_1,x_0)
=
\frac{
\hat{w}(x_1)
}{
\hat{w}(x_0)
}
=
\exp
\left[
(x_1-x_0)\hat{\beta}
\right].
$$

For two otherwise identical locations differing by one unit in covariate $x_j$,

$$
\operatorname{RSS}
=
\exp(\hat{\beta}_j).
$$

Values greater than one indicate greater relative selection for the first condition; values below one indicate lower relative selection. These contrasts remain conditional upon the other model terms, the available domain and the scale of analysis.

Interactions, transformations and nonlinear effects should be interpreted through explicit contrasts rather than through isolated coefficients. A mapped RSF surface,

$$
\hat{w}\left(x(s)\right)
=
\exp\left[x(s)\hat{\beta}\right],
$$

has an arbitrary absolute scale. It may be normalised within a defined available domain to obtain

$$
\hat{f}^{u}(s)
=
\frac{
f^{A}(s)\hat{w}\left(x(s)\right)
}{
\displaystyle
\int_{\Omega}
f^{A}(r)\hat{w}\left(x(r)\right)\,dr
},
$$

but the resulting distribution is specific to that domain.

### Individuals and Autocorrelation

Telemetry studies may contain many relocations but comparatively few animals. Locations from the same individual are repeated observations of one behavioural process and should not be treated as independent population replicates.

Individual variation may be represented hierarchically:

$$
w_i(x)
=
\exp(x\beta_i),
$$

with

$$
\beta_i
\sim
\mathcal{N}(\mu,\Sigma),
$$

where $\mu$ describes the population-level response and $\Sigma$ describes variation among individuals. Random slopes are important because animals may differ in the direction or strength of selection. A random intercept alone does not represent such behavioural variation.

Successive relocations are also serially dependent. An animal's current position constrains its next position, and nearby observations commonly share similar environmental conditions. Ignoring this dependence generally leads to underestimated uncertainty.

Autocorrelation should not automatically be removed by thinning the data until successive points appear independent. Residence, return and slow movement may themselves be biologically meaningful. More defensible approaches include hierarchical models, robust standard errors, block bootstrapping, two-stage analyses and structured cross-validation.

### Model Evaluation

Evaluation should be based upon held-out data and should reflect the intended form of prediction. Interpolation within known individuals, prediction to later periods and transfer to new individuals are distinct tasks.

A commonly used measure for presence-only or use--availability predictions is the **continuous Boyce index**. Let $\hat{w}(s)$ denote the predicted RSF value in an evaluation dataset. The prediction range is divided into intervals or moving windows $I_k$. For each interval, calculate the proportion of held-out used locations,

$$
P_k
=
\frac{
\text{held-out used locations in }I_k
}{
\text{all held-out used locations}
},
$$

and the proportion of available locations,

$$
E_k
=
\frac{
\text{available locations in }I_k
}{
\text{all available locations}
}.
$$

Their ratio is

$$
R_k
=
\frac{P_k}{E_k}.
$$

The Boyce index is the Spearman rank correlation between the representative prediction value $v_k$ and the predicted-to-expected ratio:

$$
B
=
\operatorname{cor}_{\mathrm{S}}
\left(
v_k,R_k
\right).
$$

Values approaching $1$ indicate that greater predictions correspond to greater use relative to availability. Values near $0$ indicate little monotonic relationship, while negative values indicate that held-out locations occur disproportionately in areas assigned low predictions.

The Boyce index evaluates ranking rather than complete probabilistic calibration. It does not preserve the magnitude or geographic location of prediction errors and may depend upon window width, ties and sample size. It should therefore be accompanied by the full predicted-to-expected curve, observed and expected use across prediction quantiles, and spatial inspection of held-out residual patterns.

### Structured Cross-Validation

Randomly dividing relocations among folds is generally unsuitable for telemetry data. Neighbouring observations from the same movement bout may enter both the training and evaluation sets, producing information leakage and overly optimistic performance estimates.

For prediction to another period within the same animals, data should be divided into contiguous **temporal blocks**. Blocks should be sufficiently long to reduce dependence between training and evaluation observations. Where forecasting is the objective, models should be fitted to earlier periods and evaluated on later ones.

For inference to the wider population, **leave-one-individual-out cross-validation** is more appropriate. All relocations from one animal are withheld, the model is fitted to the remaining animals, and its predictions are evaluated against the held-out individual. Repeating this procedure assesses whether the estimated population response transfers to animals not represented during fitting.

Where both temporal transfer and population transfer matter, the two schemes may be combined. An outer leave-one-individual-out loop can assess generalisation to new animals, while temporally blocked validation within the training animals can guide model development.

Validation scores should be reported by individual and fold. A single pooled score may be dominated by animals contributing the greatest number of relocations.

### From Resource Selection to Step Selection

A conventional RSF usually compares every relocation with a common availability distribution, such as an individual's home range or the study area. At fine temporal scales, this assumption becomes difficult to defend. A location may lie within the home range but remain unreachable during the interval between two fixes.

Movement therefore determines availability. The locations accessible at time $t+1$ depend upon the animal's position at time $t$, the elapsed time, its movement capacity and its previous direction of travel.

A **step** is the displacement between consecutive relocations:

$$
s_t
\longrightarrow
s_{t+1}.
$$

Its length is

$$
l_t
=
\left\|
s_{t+1}-s_t
\right\|,
$$

and its turning angle $\theta_t$ is the change in bearing relative to the preceding step.

In a **Step Selection Function**, each observed step is paired with available steps beginning at the same location. Their lengths and turning angles are conventionally sampled from the empirical distributions of observed movements. Habitat conditions at the observed endpoint or along the observed path are then compared with those of the available alternatives.

Let

$$
\mathcal{C}_t
=
\left\{
s_{t+1}^{(0)},
s_{t+1}^{(1)},
\ldots,
s_{t+1}^{(K)}
\right\}
$$

denote the choice set for step $t$, where $s_{t+1}^{(0)}$ is the observed endpoint. Under conditional logistic regression,

$$
\Pr
\left(
j\mid\mathcal{C}_t
\right)
=
\frac{
\exp\left(z_{tj}\gamma\right)
}{
\displaystyle
\sum_{k=0}^{K}
\exp\left(z_{tk}\gamma\right)
}.
$$

An SSF therefore asks which characteristics distinguished the chosen step from the alternatives reachable from the same starting point.

### Integrated Step Selection

Empirical step-length and turning-angle distributions have already been shaped by habitat selection. Short observed steps, for example, may reflect intrinsically slow movement or prolonged residence in selected habitat. Sampling available movements directly from the observed distributions therefore risks confounding movement with selection.

An **integrated Step Selection Function** estimates the movement and selection processes jointly. Let $\phi$ denote the baseline movement kernel and $w$ the habitat-selection function. The transition density is

$$
p
\left(
s_{t+1}
\mid
s_t,s_{t-1}
\right)
=
\frac{
\phi
\left(
s_{t+1}
\mid
s_t,s_{t-1};\theta
\right)
w
\left(
x(s_{t+1});\beta
\right)
}{
\displaystyle
\int_{\Omega}
\phi
\left(
r
\mid
s_t,s_{t-1};\theta
\right)
w
\left(
x(r);\beta
\right)\,dr
}.
$$

The movement kernel determines which locations can be reached and with what baseline probability. The selection function then reweights those locations according to habitat. Movement and selection together determine the next relocation.

Terms such as $l$, $\log l$ and $\cos\theta$ allow parameters of step-length and turning-angle distributions to be estimated alongside habitat-selection coefficients. The resulting model separates the baseline movement process from the effects of environmental conditions more explicitly than a conventional SSF (Avgar et al., 2016).

### Interpretation Through Simulation

An SSF or iSSF coefficient describes a local contrast among alternatives available from the same starting point. For two candidates $a$ and $b$,

$$
\frac{
\Pr(a\mid\mathcal{C}_t)
}{
\Pr(b\mid\mathcal{C}_t)
}
=
\exp
\left[
\eta_{ta}-\eta_{tb}
\right].
$$

Such contrasts remain directly interpretable. The long-term consequences of the coefficients are less immediate because each selected endpoint determines the next choice set. Movement and selection therefore alter both the present choice and the locations available in the future.

Predicted utilisation distributions, residence times, crossing probabilities and home-range geometry should consequently be obtained through simulation of the complete transition process. At each step, candidate movements are drawn from the fitted movement kernel, weighted according to habitat, and sampled to produce the next location. Repetition generates trajectories whose emergent spatial patterns can be compared with held-out observations.

RSFs, SSFs and iSSFs therefore represent a progression in the treatment of availability. An RSF relates use to a broader and approximately static available distribution. An SSF conditions availability upon the animal's previous location. An iSSF jointly estimates the movement process generating those local alternatives and the habitat-selection process distinguishing among them. The appropriate framework depends upon the temporal scale of the data and the inferential question being addressed.
## Towards Bayesian Inference

The preceding models commonly proceed as though recorded locations were exact, environmental covariates were known without error, individuals differed only through random sampling, and habitat-selection relationships remained constant through time. Each assumption is convenient; none is generally true. A Bayesian analysis is not the only means of relaxing them, but it provides a coherent framework in which uncertain observations, latent ecological processes, individual heterogeneity and temporal change may be represented jointly and propagated into the final inference.

Let $\mathcal{D}$ denote the observed data, $\mathcal{Z}$ the unobserved ecological quantities from which those data arose, and $\Theta$ the model parameters. Bayesian inference proceeds from the posterior distribution

$$
p(\mathcal{Z},\Theta\mid\mathcal{D})
\propto
p(\mathcal{D}\mid\mathcal{Z},\Theta)
p(\mathcal{Z}\mid\Theta)
p(\Theta).
$$

The first term describes the observation process, the second the ecological process and the third the prior information assigned to its parameters. Rather than replacing uncertain quantities by single estimates, the posterior integrates over their plausible values. This distinction is central to three objectives of **hrHSA**: propagating uncertainty, estimating individual variation and detecting change.

## Propagating Locational and Environmental Uncertainty

Telemetry records an estimate of an animal's location rather than the location itself. Let $s_{ij}$ denote the true location of individual $i$ at observation $j$, and let $\tilde{s}_{ij}$ denote the recorded GNSS fix. A simple observation model is

$$
\tilde{s}_{ij}
\mid
s_{ij},\Sigma_{ij}
\sim
\mathcal{N}_2
\left(
s_{ij},
\Sigma_{ij}
\right),
$$

where $\Sigma_{ij}$ describes the magnitude and orientation of locational uncertainty. Other distributions may be substituted where errors are heavy-tailed, device-specific or otherwise non-Gaussian. The ecological model is then evaluated at the latent location $s_{ij}$ rather than treating $\tilde{s}_{ij}$ as exact (Jonsen et al., 2005; Patterson et al., 2008; Hooten et al., 2017).

Locational inaccuracy must be distinguished from failed fixes. A recorded fix may be spatially imprecise, whereas a failed fix produces no location at all. If fix success depends upon canopy cover, topography or other habitat features, the observed relocations constitute a biased sample of the underlying path. The observation process may therefore require both a model for positional error and a model for the probability of detection or successful acquisition (Frair et al., 2004; Nielson et al., 2009).

Environmental data are likewise imperfect. Let $x^{*}(s,t)$ denote the environmental field relevant to the animal and let $\tilde{x}_g$ denote the value represented by raster cell $g$, covering area $A_g$. A continuous raster variable may be viewed as an imperfect spatial aggregate:

$$
\tilde{x}_g
=
\frac{1}{|A_g|}
\int_{A_g}
x^{*}(s,t)\,ds
+
\epsilon_g,
$$

where $\epsilon_g$ represents measurement, interpolation or classification error. A raster cell therefore describes an average or assigned class over an area, while the animal occupies a location within it. This difference in spatial support becomes consequential where the environmental field varies substantially within cells or where locational uncertainty is large relative to raster resolution.

Raster uncertainty arises from several sources: sensor and classification error, temporal mismatch, interpolation, reprojection, resampling and the averaging of heterogeneous conditions within pixels. Grain size is not merely another random error term. It determines the spatial support over which habitat is represented and may therefore alter the ecological relationship being estimated. Uncertainty about grain should be addressed through biologically motivated multiscale models or sensitivity analyses rather than concealed within a single residual variance (Northrup et al., 2022).

A hierarchical model may combine these uncertainties schematically as

$$
\begin{aligned}
\tilde{s}_{ij}
&\sim
p\left(
\tilde{s}_{ij}\mid s_{ij},\psi_s
\right),\\
\tilde{x}
&\sim
p\left(
\tilde{x}\mid x^{*},\psi_x
\right),\\
y_{ij}
&\sim
p\left(
y_{ij}\mid s_{ij},x^{*},\beta_i
\right),
\end{aligned}
$$

where $\psi_s$ and $\psi_x$ govern locational and environmental uncertainty. The posterior distribution of $\beta_i$ then reflects uncertainty not only in sampling, but also in the positions and environmental conditions upon which inference depends.

This propagation is especially important for derived quantities. Selection surfaces, utilisation distributions and simulated trajectories should not be calculated solely from posterior mean coefficients and a single environmental raster. They may instead be generated repeatedly from posterior draws of locations, environmental fields and model parameters, producing a distribution of predictions rather than one deceptively exact map.

## Individual Variation as Biological Information

A population-level coefficient describes a central tendency; it does not imply that all individuals respond alike. Animals may differ in habitat selection because of sex, age, reproductive state, experience, physiology, social status, competitive ability or behavioural phenotype. Pooling these differences into a single coefficient may obscure opposing responses and understate uncertainty at the population level (Leclerc et al., 2016; Muff et al., 2020; Northrup et al., 2022).

Individual-specific selection coefficients may be represented hierarchically as

$$
\beta_i
=
\mu_{\beta}
+
Z_i\gamma
+
b_i,
$$

with

$$
b_i
\sim
\mathcal{N}
\left(
0,\Sigma_{\beta}
\right).
$$

Here, $\mu_{\beta}$ is the population-level mean response, $Z_i\gamma$ describes systematic differences associated with measured individual attributes, and $b_i$ represents residual variation among individuals. The covariance matrix $\Sigma_{\beta}$ quantifies both the magnitude of individual variation and correlations among selection responses.

This model partially pools information across animals. Individuals with abundant and informative data retain strongly individual estimates, whereas poorly sampled individuals are drawn towards the population distribution. Partial pooling avoids the two extremes of treating every relocation as an independent population replicate and fitting entirely separate models whose imprecise estimates are subsequently treated as exact.

Individual variation also changes population-level prediction. Because the selection function is nonlinear,

$$
\operatorname{E}_i
\left[
\exp(x\beta_i)
\right]
\neq
\exp
\left[
x\operatorname{E}_i(\beta_i)
\right].
$$

The prediction obtained from the mean individual is therefore not generally equal to the mean prediction across individuals. Population-level maps should integrate over the estimated distribution of individual responses rather than merely substitute $\mu_{\beta}$ into the selection function.

Among-individual variation is not statistical debris around a population mean. It is a property of the population and may reveal individual specialisation, alternative behavioural tactics or differences in environmental sensitivity. Phenotypic variation is also the material upon which natural selection can act. Variation in estimated habitat-selection behaviour is not, by itself, evidence of adaptive potential: an evolutionary response additionally requires that differences are sufficiently persistent, heritable and associated with fitness. Hierarchical habitat-selection models can establish and quantify behavioural variation, while pedigrees, genomic information, repeated observations and demographic data are required to distinguish its genetic, environmental and fitness-related components (Leclerc et al., 2016; Gervais et al., 2022).

## Detecting Change

Conventional selection models assume that the coefficient vector $\beta$ remains constant over the period of inference. Yet selection may vary with season, life-history stage, population density, resource availability, disturbance, learning or climatic conditions. In a changing environment, it is therefore insufficient to ask only which habitats are selected on average. We must also ask whether, when and among whom the relationship is changing.

A time-varying selection function may be written as

$$
w_{i,t}(x)
=
\exp
\left(
x\beta_{i,t}
\right).
$$

Where change is expected to be gradual, the coefficients may evolve according to a stochastic process such as

$$
\beta_{i,t}
\sim
\mathcal{N}
\left(
\beta_{i,t-1},
Q
\right),
$$

where $Q$ controls the rate and covariance of temporal change. Smaller values of $Q$ imply comparatively stable selection, whereas larger values permit more rapid variation. Alternative models may impose seasonal periodicity, smooth temporal trends or dependence upon measured environmental drivers.

Where an abrupt transition is expected, a change-point model may instead be used:

$$
\beta_{i,t}
=
\begin{cases}
\beta_i^{(1)}, & t < \tau_i,\\
\beta_i^{(2)}, & t \geq \tau_i,
\end{cases}
$$

where $\tau_i$ is the unknown time of change. Bayesian inference yields a posterior distribution for $\tau_i$ rather than forcing the transition to coincide with an analyst-defined season or calendar date.

Temporal change and individual variation may be separated through

$$
\beta_{i,t}
=
\mu_t
+
b_i
+
u_{i,t},
$$

where $\mu_t$ is the population-wide trajectory, $b_i$ is a persistent individual deviation and $u_{i,t}$ describes within-individual change. This distinction separates population-wide redistribution from differences in individual timing and response. It also prevents changes in the composition of the sampled population from being mistaken for behavioural change within individuals.

Posterior distributions permit direct statements concerning change. For a coefficient $\beta_j$, one may calculate

$$
\Pr
\left(
\beta_{j,t_2}
-
\beta_{j,t_1}
>
0
\mid
\mathcal{D}
\right),
$$

or the posterior probability that a change exceeds a biologically meaningful threshold. The result describes the magnitude, direction and uncertainty of change rather than reducing inference to a sequence of separate significance tests.

This capacity is particularly relevant under global change. Changes in climate, land use and human disturbance may alter both the distribution of available habitat and the manner in which animals respond to it. A dynamic model can distinguish a changing environmental landscape from a changing selection relationship, provided that both used and available conditions are represented through time. It may reveal gradual adjustment, abrupt displacement, increased variability or divergent responses among individuals.

Temporal change in a coefficient does not, however, identify its cause. Apparent non-stationarity may arise from season, reproduction, ageing, population density, behavioural state, changing availability or alteration of the observation process. Attribution to global change requires explicit environmental hypotheses, suitable temporal replication and, where possible, comparison across populations or landscapes. Dynamic Bayesian models expose change and quantify its uncertainty; they do not replace ecological reasoning.

## Posterior Prediction and Model Evaluation

The principal practical advantage of the Bayesian formulation is that uncertainty can be carried into every derived prediction. Let $\Theta$ collect the parameters and latent states of the fitted model. The posterior predictive distribution is

$$
p
\left(
\tilde{\mathcal{D}}
\mid
\mathcal{D}
\right)
=
\int
p
\left(
\tilde{\mathcal{D}}
\mid
\Theta
\right)
p
\left(
\Theta
\mid
\mathcal{D}
\right)
\,d\Theta.
$$

Each posterior draw represents one plausible state of the ecological system. Repeated prediction or simulation therefore propagates uncertainty in locations, environmental covariates, individual responses and temporal dynamics into selection maps, utilisation distributions and simulated movement paths.

Bayesian inference does not remove the need for validation. Priors should be examined through prior-predictive simulation, fitted models through posterior-predictive checks, and predictive performance through the temporally blocked and leave-one-individual-out procedures described in the preceding chapter. Weak identifiability, an inappropriate availability distribution or an inadequate ecological model cannot be repaired by increasingly elaborate computation.

The purpose of the Bayesian extension is therefore to bring the inferential model into closer correspondence with the ecological system: locations and environments are observed imperfectly, populations consist of heterogeneous individuals, and selection may change through time. Representing these processes jointly permits uncertainty to remain visible, individual variation to become an object of inference and ecological change to be detected rather than averaged away.

## Principal References

Auger-Méthé, M., Newman, K., Cole, D., Empacher, F., Gryba, R., King, A. A., Leos-Barajas, V., Mills Flemming, J., Nielsen, A., Petris, G. & Thomas, L. (2021). A guide to state-space modeling of ecological time series. *Ecological Monographs*, **91**, e01470.

Dejeante, R., Valeix, M. & Chamaillé-Jammes, S. (2024). Time-varying habitat selection analysis: a model and applications for studying diel, seasonal, and post-release changes. *Ecology*, **105**, e4233.

Frair, J. L., Nielsen, S. E., Merrill, E. H., Lele, S. R., Boyce, M. S., Munro, R. H. M., Stenhouse, G. B. & Beyer, H. L. (2004). Removing GPS collar bias in habitat selection studies. *Journal of Applied Ecology*, **41**, 201–212.

Gervais, L., Morellet, N., David, I., Hewison, M., Réale, D., Goulard, M., Chaval, Y., Lourtet, B., Cargnelutti, B., Merlet, J., Quéméré, E. & Pujol, B. (2022). Quantifying heritability and estimating evolutionary potential in the wild when individuals that share genes also share environments. *Journal of Animal Ecology*, **91**, 1239–1250.

Hooten, M. B., Johnson, D. S., McClintock, B. T. & Morales, J. M. (2017). *Animal Movement: Statistical Models for Telemetry Data*. Boca Raton: CRC Press.

Jonsen, I. D., Flemming, J. M. & Myers, R. A. (2005). Robust state-space modeling of animal movement data. *Ecology*, **86**, 2874–2880.

Leclerc, M., Vander Wal, E., Zedrosser, A., Swenson, J. E., Kindberg, J. & Pelletier, F. (2016). Quantifying consistent individual differences in habitat selection. *Oecologia*, **180**, 697–705.

Muff, S., Signer, J. & Fieberg, J. (2020). Accounting for individual-specific variation in habitat-selection studies: efficient estimation of mixed-effects models using Bayesian or frequentist computation. *Journal of Animal Ecology*, **89**, 80–92.

Nielson, R. M., Manly, B. F. J., McDonald, L. L., Sawyer, H. & McDonald, T. L. (2009). Estimating habitat selection when GPS fix success is less than 100%. *Ecology*, **90**, 2956–2962.

Northrup, J. M., Vander Wal, E., Bonar, M., Fieberg, J., Laforge, M. P., Leclerc, M., Prokopenko, C. M. & Gerber, B. D. (2022). Conceptual and methodological advances in habitat-selection modeling: guidelines for ecology and evolution. *Ecological Applications*, **32**, e02470.

Patterson, T. A., Thomas, L., Wilcox, C., Ovaskainen, O. & Matthiopoulos, J. (2008). State-space models of individual animal movement. *Trends in Ecology & Evolution*, **23**, 87–94.

Aarts, G., Fieberg, J. & Matthiopoulos, J. (2012). Comparative interpretation of count, presence--absence and point methods for species distribution models. *Methods in Ecology and Evolution*, **3**, 177--187.

Avgar, T., Potts, J. R., Lewis, M. A. & Boyce, M. S. (2016). Integrated step selection analysis: bridging the gap between resource selection and animal movement. *Methods in Ecology and Evolution*, **7**, 619--630.

Boyce, M. S., Vernier, P. R., Nielsen, S. E. & Schmiegelow, F. K. A. (2002). Evaluating Resource Selection Functions. *Ecological Modelling*, **157**, 281--300.

Fieberg, J., Matthiopoulos, J., Hebblewhite, M., Boyce, M. S. & Frair, J. L. (2010). Correlation and studies of habitat selection: problem, red herring or opportunity? *Philosophical Transactions of the Royal Society B*, **365**, 2233--2244.

Fieberg, J., Signer, J., Smith, B. & Avgar, T. (2021). A “how to” guide for interpreting parameters in habitat-selection analyses. *Journal of Animal Ecology*, **90**, 1027--1043.

Forester, J. D., Im, H. K. & Rathouz, P. J. (2009). Accounting for animal movement in estimation of Resource Selection Functions: sampling and data analysis. *Ecology*, **90**, 3554--3565.

Fortin, D., Beyer, H. L., Boyce, M. S., Smith, D. W., Duchesne, T. & Mao, J. S. (2005). Wolves influence elk movements: behavior shapes a trophic cascade in Yellowstone National Park. *Ecology*, **86**, 1320--1330.

Hirzel, A. H., Le Lay, G., Helfer, V., Randin, C. & Guisan, A. (2006). Evaluating the ability of habitat-suitability models to predict species presences. *Ecological Modelling*, **199**, 142--152.

Johnson, C. J., Nielsen, S. E., Merrill, E. H., McDonald, T. L. & Boyce, M. S. (2006). Resource Selection Functions based on use--availability data: theoretical motivation and evaluation methods. *Journal of Wildlife Management*, **70**, 347--357.

Northrup, J. M., Hooten, M. B., Anderson, C. R. & Wittemyer, G. (2013). Practical guidance on characterizing availability in Resource Selection Functions under a use--availability design. *Ecology*, **94**, 1456--1463.

Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, **40**, 913--929.

Warton, D. I. & Shepherd, L. C. (2010). Poisson point-process models solve the “pseudo-absence problem” for presence-only data in ecology. *The Annals of Applied Statistics*, **4**, 1383--1402.
