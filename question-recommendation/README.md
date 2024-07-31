# KTbasedRecSys


## OVERALL
A simple Recommendation system guided by knowledge tracing for generating student quiz paper.
- Please read papers related to "knowledge tracing" and "knowledge graph"
- this project is used for a small company, so if you want to reuese, esliminate code connecting to DB


## Component

- KT model 

# Project Organization
------------
    ├── README.md          <- The top-level README for developers using this project.
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── deploy             <- **CODE for deploy solution on AWS services**
    │   ├── __old_         <- contain old source code of pals version runing < 2/2024 
    │   │
    │   ├── ktmodel-standard        <- code to deploy KT model for PALS on aws lambda
    │   ├── recsys-standard         <- code to deploy main RecSys of PALS on aws lambda
    │   │
    │   └── update-FI  <- Scripts to update FI value in pals table
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── src                <- **Source code for training AI model.**
    │   ├── kgbuilding     <- contain ipynb file to construct KG for PALS
    │   │
    │   ├── ktmodel        <- code used to train KT model for each subject in PALS db
    │   │
    │   │
    │   └── update-FI  <- Scripts to update FI value in pals table
    |....



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


# Cite

```cite
@misc{2208.12615,
   Author = {Unggi Lee and Yonghyun Park and Yujin Kim and Seongyune Choi and Hyeoncheol Kim},
   Title = {MonaCoBERT: Monotonic attention based ConvBERT for Knowledge Tracing},
   Year = {2022},
   Eprint = {arXiv:2208.12615},
}



```
