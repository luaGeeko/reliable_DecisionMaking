### NMA_Project : Decision Making

This repository will be used for the NMA project collaboration. Initial outline steps have been defined and necessary links to documents will be added.

#### Steinmetz dataset

Dataset link https://osf.io/agvxh . The dataset has three parts on the site. It can be downloaded from https://drive.google.com/drive/folders/1FTCIIrRKIzEg0nIetOHWF34zKTrFwIfT?usp=sharing.

To help with laoding of dataset and initial analysis overview this colab notebook can be referred to https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/load_steinmetz_decisions.ipynb copy of which is available in repo.

#### Instructions to load data
1. First download the data from the google drive and place it under the folder ```dataset ```
2. ```load_data.py``` module will load the data with 40 sessions from 10 mice
3. ```split_test_and_training_data.py``` module will split the data into training and test sets.
4. ```generate_dummy_data.py``` module will simulate choice neurons and neurons that do not have activity that correlates with the choice. This will be used to test the GLM.
  
 
#### Some Helpful Git commands
1. To create a new branch from the master ```git branch new-branch```
2. List all the branches in repository ```git branch -a```
3. Checkout to different branch ``` git checkout branch_name``` example: ```git checkout master```
4. To pull new changes ```git pull```. Before pushing new code make sure your repository is in sync
5. Once changes done to the code to push it to the repository ```git push```
6. Make sure you keep your branches in sync with the master branch, follow these steps:
     1. first move to master branch ``` git checkout master ``` 
     2. pull the updates   ```git pull```
     3. now move back to your branch ```git checkout your-branch-name```
     4. merge with master ```git merge master``` 
