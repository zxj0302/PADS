- [X] 分类论文
- [X] 整理comments
- [X] 整理mitigation strategy
- [ ] 应用新的polarizaton measurement
- [ ] Model-independent suggestions or mitigation strategies

TODO:

- [X] Cite the mitigation part in the related papers
- [X] Make the reweighted nodes the same number
- [X] In connection addition, also show the related DSP algorithms
- [X] Use the opinion dynamic model that considers the confirmation bias
- [X] Research about the mitigation strategy in the related papers
- [X] Scalibility
- [ ] Use peeling instead of adding
- [ ] Use heatmap to show the opinion changes after adding connections
- [ ] Change the loss function for GIN

TODO:

- [X] Hypothesis about the ECC existence
- [X] Coreelation between the diffusion model and the mitigation
- [X] **Properties** of the ECCs
- [X] Scalability of the algorithm

TODO:

- [X] Reachability within the same ECC or between different ECCs
- [X] Draw one whole graph for adding connections and Reweighting alone and both
- [X] Dynamic condition
- [X] Change the number of labels
- [X] Revise the paper(change 'polarity' to 'leaning')
- [X] Collate the code repository
- [ ] NP-H prove, and prove the unbounded below precision of peeling alg

TODO:

- [X] Add connections alone
- [X] Reweight alone
- [X] At the same time
- [X] Draw the distance-opinion graph
- [X] Reachability within the same ECC or between different ECCs
- [X] Results by changing theta
- [ ] Use labels that does not related to polarity
- [X] The curve of f(S) values under each operation
- [ ] Explain the settings used in diffusion model, Flowless, and PADS

TODO:

- [X] 准确的最密子图找到的结果，以及时间
- [ ] 带负边的最密子图找到的结果，以及时间
- [ ] 实现算法和Greedy++以及准确的最密子图对比, peeling and gnn convolution combined

Problems:

- [ ] By using opinion dynamics, the polarity variance is decreasing, which may not be the case in real world. I tried FJ model, HK model, and other models, but the results are the same. Maybe they are too idealized and cannot reflect the real world. In the real world, users post their opinions with different frequencies, and have different influence on others, so the opinion dynamics should be more complex(Maybe we can publish a paper for this).
- [X] If we add edges between users in ECCs, the result is worse than adding edges between high-degree users. So how can we find the best way to demonstrate the usefulness of ECCs?
