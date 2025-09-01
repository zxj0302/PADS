# On Echo Chamber Cores and Their Detection

This is code repo for the paper "On Echo Chamber Cores and Their Detection". The abstract for the paper is as follows:

Echo chambers on social media cluster like-minded users, amplifying shared beliefs while isolating opposing viewpoints. Within these chambers, we introduce the novel concept of an *Echo Chamber Core (ECC)*—a subset of highly active, ideologically radical, and like-minded users who exert disproportionate influence on both internal and external discourse. To the best of our knowledge, this is the first work to formally define, formulate, and detect such cores. Identifying ECCs is crucial for understanding their role in polarization and designing effective countermeasures against misinformation.

We propose a new method for ECC detection by modeling them as *Purity-Aware Densest Subgraphs (PADS)*. Our approach integrates two key properties:  

1. *density* — capturing the high connectivity among core users, and  
2. *purity* — reflecting the homogeneity of their opinions and polarizations.  

We prove that this problem is NP-hard. To address this challenge, we develop an efficient greedy algorithm that jointly optimizes for density and purity. To evaluate the nature and impact of ECCs, we conduct experiments on real-world social media datasets. Our results demonstrate that the detected ECCs are densely connected, highly homophilic, and strongly polarized, validating the effectiveness of our detection framework and offering new insights into the structural and semantic properties of echo chambers.

Beyond detection, we propose an ECC-driven mitigation strategy to address the structural effects of echo chambers and support more balanced information ecosystems. By formalizing the ECC concept and presenting both detection and intervention methods, this work provides a foundation for targeted, data-driven responses to online polarization.

## About This Code Repo

### Structure and Organization

The repository is organized as follows:

```text
.
├── input/              # Pre-processed input datasets put in input/datasets/static, can find raw files in according paper
│                       # Note that, the 'VoterFraud2020' dataset is put in [1] due to its large size
│                       # Raw tweets are not provided/re-shared according the terms of service of X (Twitter)
├── others/             # Some TODO files and backup files, not uploaded here
├── output/             # Generated outputs (results, processed data, figures, etc.)
├── Related_Reps/       # Related repositories/implemented & optimized baselines (source & compiled files) used for comparison. Too large, thus put in [2]
├── src/                # Source code and implementation scripts (compiled pads.cpp are in [2])
├── .gitignore          # Git ignore file to exclude unnecessary files
├── environment.yml     # Conda env for running code
├── figures.ipynb       # Notebook for generating and visualizing figures used in paper and results analysis
├── main.ipynb          # Main notebook for running core analysis/workflow
├── mitigation.ipynb    # Notebook focusing on experiments about mitigation strategies
└── README.md           # Project documentation (this file)
```

### Compile and Run

1. Jupyter files:
    * recover conda env

    ```bash
    conda env create -f environment.yml
    ```

    * run main.ipynb/mitigation.ipynb/figures.ipynb, code logic with comments are clear in these files and omitted here

2. Python files:
    * Same as above, check specific .py files for their usage
3. CPP files (in Related_Reps/):
    * All files compiled (to .exe file, thus can be directly run on Windows)
    * For recompilation to apply changes or using them on other systems (e.g. Linux, MacOS):

    ```bash
    cd path/to/project
    # for projects in Related_Reps/ with a makefile
    make
    # for projects in Related_Reps/ with a CMakeLists.txt
    mkdir build
    cd build
    cmake ..
    ```  

    * More detailed compilation process and commands are in README in each projects under Related_Reps/

## Discussions That Are Not in the Paper Due to Space Limitation

### Other Baselines for ECC Detection

In addition to the aforementioned baselines, we also compared our method with unweighted MaxFlow, DITH~\cite{fazzone2022discovering}, EIGENSIGN~\cite{bonchi2019discovering}, Surprise~\cite{marchese2022detecting}, KM-config~\cite{kojaku2018core}. However, these approaches are not discussed in this paper due to the following reasons. For unweighted MaxFlow, its results are consistently dominated by weighted MaxFlow. DITH, despite our extensive experimentation with different parameter settings ($\lambda_1, \lambda_2$ as defined in their paper), edge weight configurations (e.g., uniform weights or weights based on leaning similarity), and implementations (both theirs and ours), fails to identify meaningful ECCs. The outputs are either highly overlapping ECCs or sparse subgraphs with too few nodes, lacking the distinctive characteristics of ECCs. This behavior likely arises from its sensitivity to parameter settings: small $\lambda$ values result in the densest subgraph across the entire graph (thus highly overlapped), while large $\lambda$ values prioritize distance from repulsers and proximity to attractors at the expense of density. Another potential limitation lies in the selection of authorities. Without access to user identification information, assigning appropriate authorities becomes challenging. EIGENSIGN optimizes both intra-group positive edges and inter-group negative edges. However, in polarized social networks, especially ECCs and retweet networks that primarily represent endorsement relationships, interactions between users with opposing views are minimal. As a result, EIGENSIGN often produces a single dense subgraph (always in the densest part across the whole graph), while the opposing group contains too few nodes due to its penalization of group size. We evaluated Surprise and KM-config under both binary and continuous edge weight settings. However, on most datasets, Surprise tends to produce subgraphs that are larger but lack sufficient density or purity. And the KM-config algorithm also consistently produce cores that are either insufficiently dense or overly large. This outcome may stem from limitations in KM-config’s quality function, which, similar to modularity, struggles to detect small communities, as discussed in the original paper. Although these methods do not perform well in our specific cases, they may be better suited for other scenarios, such as those presented in their original studies. For example, we find that the cores detected by KM-config always have larger conductance, which may be useful when the outgoing connection of the core is a focus.

### Mitigation: Further Discussions

(Put after discussion about mitigation in related work): A key limitation of these opinion dynamics-based methods is their heavy reliance on the simplified Friedkin--Johnsen model. This model assumes a fixed listening structure~\cite{chen2022polarizing} and static user stubbornness, which may restrict its capacity to capture dynamic network adaptation. Furthermore, highly idealized strategies, such as directly changing innate opinions, are often impractical for real-world social network applications.

(Put in Section 6.5): While the strategies discussed above illustrate how ECC detection can suppress or mitigate polarization in echo chambers, we now explore feasibility and limitations. Although cross-group interaction promotion is effective in reducing polarization, achieving a low degree of polarization, confirmation bias often hinders establishing effective discourse and connections, particularly when users act strategically for profit or other motives. Moreover, exposure to opposing political views may trigger backfire effects, potentially exacerbating polarization in some cases ~\cite{bail2018exposure}. With our strategy, we can avoid the backfire effects to some degree, as we favor to connect users with moderately opposing opinions. In practice, the recommendation system can suggest posts that slightly challenge a user's views to promote cross-group interaction. For instance, if a Twitter user is strongly anti-vaccine due to conspiracy theory beliefs, the system could recommend tweets with neutral to slightly pro-vaccine perspectives to foster open-mindedness and reduce backfire effects.

## Change Logs

[2025-09-02] First version README.md and uploaded to repo.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Links and Reference

1. [Drive for VoterFraud2020](https://drive.google.com/file/d/1YD02L37M-3PF2ru0xSbhi8zw0VtReTSS/view?usp=sharing)
2. [Drive for Related_Reps](https://drive.google.com/file/d/1fF9nXSYKVnfeegq3s1IZeJJDVnG3hAAJ/view?usp=sharing)
