CML progect
===========
Project description
-------------------
In this project we'll compare ML models across two different Git branches of a project and we'll do it in a continuous integration system (GitHub Actions) using dvc.

**We'll cover:**
- How to create dvc pipeline
- How to create workflow for github actions

**For what:**
- Thanks to the DVC pipeline, we will be able to change our source data, input parameters and quickly reproduce calculations.
- Thanks to the CML pipeline, we will be able to compare the results (metrics) of two different branches.

**Input information:**
- data - data frame about thyroid disease
- model - Logistic regression
- 2 branches - unbalanced and balanced data frame

**Content:**
- requirements.txt - pipeline's element for installing necessary libraries
- src - folder for python scripts
- data - folder for dvc link to the data
- config - file for setting remote storage for our data
- dvc.yaml - dvc pipeline
- params.yaml - file which can be changed in the process of pipeline or manually
- unit_tests.py - file for testing scripts
- test.yaml - CICD pipeline

**Process:**
To begin with, I set up the dvc (dvc init). Then I added my dataframe for tracking dvc. I put my data in gdrive for sharing (config file).
DVC pipeline includes 3 stages - preparation, evaluation, feature importance. In each stage pipelile get python script, parameters and give out files.
Further DVC reproduction, running experiments, stylistic and logical code verification, testing happens into test.yaml. The results will be presented in a markdown format after creating pull request for necessary branches and after git push in branch

**Detailes:**

**_config file_** - Access to data on gdrive goes through google service account
                    [Useful link](https://dvc.org/doc/user-guide/setup-google-drive-remote)

**_dvc.yaml_**
cache: false - above specifies that metrics.json is not tracked or cached by DVC (-M option of dvc stage add). These metrics files are normally committed with Git instead

**_test.yaml_**
always placed in the same place .github/workflows

- _on: [pull_request]_ – it’s a trigger for starting pipeline

- _jobs:
  _lint:_ – job’s name - A job is a set of steps that a workflow should execute on the same runner. This could either be a shell script or an action. Steps are executed in order in the same runner and are dependent on each other.

- runs-on: ubuntu-latest - Runner. This indicates the server the job should run on. It could be Ubuntu Linux, Microsoft Windows, or macOS
- This action checks-out your repository under $GITHUB_WORKSPACE, so your workflow can access it.
                            Only a single commit is fetched by default, for the ref/SHA that triggered the workflow. Set fetch-depth: 0 to fetch all history for all branches and tags. Refer here to learn which commit $GITHUB_SHA points to for different events.
                          The auth token is persisted in the local git config. This enables your scripts to run authenticated git commands. The token is removed during post-job cleanup. Set persist-credentials: false to opt-out.
```yaml
    steps:
      - uses: actions/checkout@v2
        with:
          fetch-depth: 0
  ```
- Setup node package manager
```yaml
        uses: action/setup-node@v1
```

- Uses: psf/black@stable – black formatting is integrated in github actions
- Env:
repo_token: ${{ secrets.GITHUB_TOKEN }}  - At the start of each workflow run, GitHub automatically creates a unique GITHUB_TOKEN secret to use in your workflow. 
You can use the GITHUB_TOKEN to authenticate in a workflow run.
When you enable GitHub Actions, GitHub installs a GitHub App on your repository. 
The GITHUB_TOKEN secret is a GitHub App installation access token. 
You can use the installation access token to authenticate on behalf of the GitHub App installed on your repository. 
The token's permissions are limited to the repository that contains your workflow. 
Before each job begins, GitHub fetches an installation access token for the job. 
The GITHUB_TOKEN expires when a job finishes or after a maximum of 24 hours.
[automatic-token-authentication](https://docs.github.com/en/actions/security-guides/automatic-token-authentication)

- GDRIVE_CREDENTIALS_DATA: ${{ secrets.GDRIVE_CREDENTIALS_DATA }} - necessary permitions for DVC to complete the connection with gdrive
- npm i -g @dvcorg/cml - package manager

- sudo apt-get install -y \
            libcairo2-dev libfontconfig-dev \
            libgif-dev libjpeg-dev libpango1.0-dev librsvg2-dev
          npm install -g vega-cli vega-lite – package for drawing plots
          
- git config user.name 'Tatiana'
          git config user.email '77466010+TanyaLiving@users.noreply.github.com' – setting users attributes with commits in git
          
- dvc exp run -S n_neighbors=5 -S random_state=24 -S n_splits_cv=10 - Setting parameters for experiments and run them
- dvc exp show, dvc exp diff - Comparing metrics and parameters of all experiments and fixing changes in report file.
- cml-publish FI_plot_compensated_hypothyroid.png --md >> report.md - drawing graphs 
- cml-send-comment report.md - output to the screen report.md 
- python unit_tests.py - checking the functionality of the code
- pylint --fail-under 9 src/train_model.py - Pylint is a tool that checks for errors in Python code, tries to enforce a coding standard and looks for code smells. The threshold = 9 out of 10. If code evaluation will be below pipeline will failed.

