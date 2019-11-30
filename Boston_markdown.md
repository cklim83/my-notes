---
layout: default
title: Boston Run
nav_order: 7
---

# Predicting Boston Housing Prices

## Using XGBoost in SageMaker (Batch Transform)

_Deep Learning Nanodegree Program | Deployment_

---

As an introduction to using SageMaker's Low Level Python API we will look at a relatively simple problem. Namely, we will use the [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) to predict the median value of a home in the area of Boston Mass.

The documentation reference for the API used in this notebook is the [SageMaker Developer's Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/)

## General Outline

Typically, when using a notebook instance with SageMaker, you will proceed through the following steps. Of course, not every step will need to be done with each project. Also, there is quite a lot of room for variation in many of the steps, as you will see throughout these lessons.

1. Download or otherwise retrieve the data.
2. Process / Prepare the data.
3. Upload the processed data to S3.
4. Train a chosen model.
5. Test the trained model (typically using a batch transform job).
6. Deploy the trained model.
7. Use the deployed model.

In this notebook we will only be covering steps 1 through 5 as we just want to get a feel for using SageMaker. In later notebooks we will talk about deploying a trained model in much more detail.

## Step 0: Setting up the notebook

We begin by setting up all of the necessary bits required to run our notebook. To start that means loading all of the Python modules we will need.


```python
%matplotlib inline

import os

import time
from time import gmtime, strftime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
import sklearn.model_selection
```

In addition to the modules above, we need to import the various bits of SageMaker that we will be using. 


```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

# This is an object that represents the SageMaker session that we are currently operating in. This
# object contains some useful information that we will need to access later such as our region.
session = sagemaker.Session()

# This is an object that represents the IAM role that we are currently assigned. When we construct
# and launch the training job later we will need to tell it what IAM role it should have. Since our
# use case is relatively simple we will simply assign the training job the role we currently have.
role = get_execution_role()
```

## Step 1: Downloading the data

Fortunately, this dataset can be retrieved using sklearn and so this step is relatively straightforward.


```python
boston = load_boston()
```

## Step 2: Preparing and splitting the data

Given that this is clean tabular data, we don't need to do any processing. However, we do need to split the rows in the dataset up into train, test and validation sets.


```python
# First we package up the input data and the target variable (the median value) as pandas dataframes. This
# will make saving the data to a file a little easier later on.

X_bos_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
Y_bos_pd = pd.DataFrame(boston.target)

# We split the dataset into 2/3 training and 1/3 testing sets.
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_bos_pd, Y_bos_pd, test_size=0.33)

# Then we split the training set further into 2/3 training and 1/3 validation sets.
X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.33)
```

## Step 3: Uploading the data files to S3

When a training job is constructed using SageMaker, a container is executed which performs the training operation. This container is given access to data that is stored in S3. This means that we need to upload the data we want to use for training to S3. In addition, when we perform a batch transform job, SageMaker expects the input data to be stored on S3. We can use the SageMaker API to do this and hide some of the details.

### Save the data locally

First we need to create the test, train and validation csv files which we will then upload to S3.


```python
# This is our local data directory. We need to make sure that it exists.
data_dir = '../data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
```


```python
# We use pandas to save our test, train and validation data to csv files. Note that we make sure not to include header
# information or an index as this is required by the built in algorithms provided by Amazon. Also, for the train and
# validation data, it is assumed that the first entry in each row is the target variable.

X_test.to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)

pd.concat([Y_val, X_val], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([Y_train, X_train], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
```

### Upload to S3

Since we are currently running inside of a SageMaker session, we can use the object which represents this session to upload our data to the 'default' S3 bucket. Note that it is good practice to provide a custom prefix (essentially an S3 folder) to make sure that you don't accidentally interfere with data uploaded from some other notebook or project.


```python
prefix = 'boston-xgboost-LL'

test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

## Step 4: Train and construct the XGBoost model

Now that we have the training and validation data uploaded to S3, we can construct a training job for our XGBoost model and build the model itself.

### Set up the training job

First, we will set up and execute a training job for our model. To do this we need to specify some information that SageMaker will use to set up and properly execute the computation. For additional documentation on constructing a training job, see the [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTrainingJob.html) reference.


```python
# We will need to know the name of the container that we want to use for training. SageMaker provides
# a nice utility method to construct this for us.
container = get_image_uri(session.boto_region_name, 'xgboost')

# We now specify the parameters we wish to use for our training job
training_params = {}

# We need to specify the permissions that this training job will have. For our purposes we can use
# the same permissions that our current SageMaker session has.
training_params['RoleArn'] = role

# Here we describe the algorithm we wish to use. The most important part is the container which
# contains the training code.
training_params['AlgorithmSpecification'] = {
    "TrainingImage": container,
    "TrainingInputMode": "File"
}

# We also need to say where we would like the resulting model artifacts stored.
training_params['OutputDataConfig'] = {
    "S3OutputPath": "s3://" + session.default_bucket() + "/" + prefix + "/output"
}

# We also need to set some parameters for the training job itself. Namely we need to describe what sort of
# compute instance we wish to use along with a stopping condition to handle the case that there is
# some sort of error and the training script doesn't terminate.
training_params['ResourceConfig'] = {
    "InstanceCount": 1,
    "InstanceType": "ml.m4.xlarge",
    "VolumeSizeInGB": 5
}
    
training_params['StoppingCondition'] = {
    "MaxRuntimeInSeconds": 86400
}

# Next we set the algorithm specific hyperparameters. You may wish to change these to see what effect
# there is on the resulting model.
training_params['HyperParameters'] = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.8",
    "objective": "reg:linear",
    "early_stopping_rounds": "10",
    "num_round": "200"
}

# Now we need to tell SageMaker where the data should be retrieved from.
training_params['InputDataConfig'] = [
    {
        "ChannelName": "train",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": train_location,
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        "ContentType": "csv",
        "CompressionType": "None"
    },
    {
        "ChannelName": "validation",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": val_location,
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        "ContentType": "csv",
        "CompressionType": "None"
    }
]
```

    WARNING:root:There is a more up to date SageMaker XGBoost image.To use the newer image, please set 'repo_version'='0.90-1. For example:
    	get_image_uri(region, 'xgboost', '0.90-1').


### Execute the training job

Now that we've built the dictionary object containing the training job parameters, we can ask SageMaker to execute the job.


```python
# First we need to choose a training job name. This is useful for if we want to recall information about our
# training job at a later date. Note that SageMaker requires a training job name and that the name needs to
# be unique, which we accomplish by appending the current timestamp.
training_job_name = "boston-xgboost-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
training_params['TrainingJobName'] = training_job_name

# And now we ask SageMaker to create (and execute) the training job
training_job = session.sagemaker_client.create_training_job(**training_params)
```

The training job has now been created by SageMaker and is currently running. Since we need the output of the training job, we may wish to wait until it has finished. We can do so by asking SageMaker to output the logs generated by the training job and continue doing so until the training job terminates.


```python
session.logs_for_job(training_job_name, wait=True)
```

    2019-11-30 10:16:05 Starting - Starting the training job...
    2019-11-30 10:16:07 Starting - Launching requested ML instances......
    2019-11-30 10:17:08 Starting - Preparing the instances for training...
    2019-11-30 10:17:59 Downloading - Downloading input data...
    2019-11-30 10:18:40 Training - Training image download completed. Training in progress.
    2019-11-30 10:18:40 Uploading - Uploading generated training model.[31mArguments: train[0m
    [31m[2019-11-30:10:18:35:INFO] Running standalone xgboost training.[0m
    [31m[2019-11-30:10:18:35:INFO] File size need to be processed in the node: 0.02mb. Available memory size in the node: 8522.68mb[0m
    [31m[2019-11-30:10:18:35:INFO] Determined delimiter of CSV input is ','[0m
    [31m[10:18:35] S3DistributionType set as FullyReplicated[0m
    [31m[10:18:35] 227x13 matrix with 2951 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
    [31m[2019-11-30:10:18:35:INFO] Determined delimiter of CSV input is ','[0m
    [31m[10:18:35] S3DistributionType set as FullyReplicated[0m
    [31m[10:18:35] 112x13 matrix with 1456 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 8 extra nodes, 0 pruned nodes, max_depth=3[0m
    [31m[0]#011train-rmse:18.8786#011validation-rmse:18.8742[0m
    [31mMultiple eval metrics have been passed: 'validation-rmse' will be used for early stopping.
    [0m
    [31mWill train until validation-rmse hasn't improved in 10 rounds.[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 8 extra nodes, 2 pruned nodes, max_depth=3[0m
    [31m[1]#011train-rmse:15.4126#011validation-rmse:15.5352[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 0 pruned nodes, max_depth=3[0m
    [31m[2]#011train-rmse:12.6695#011validation-rmse:13.0283[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 0 pruned nodes, max_depth=4[0m
    [31m[3]#011train-rmse:10.4345#011validation-rmse:11.0021[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[4]#011train-rmse:8.62027#011validation-rmse:9.48191[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[5]#011train-rmse:7.19388#011validation-rmse:8.27781[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[6]#011train-rmse:6.04045#011validation-rmse:7.24294[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[7]#011train-rmse:5.1549#011validation-rmse:6.57285[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[8]#011train-rmse:4.43242#011validation-rmse:6.01496[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[9]#011train-rmse:3.90184#011validation-rmse:5.60104[0m
    [31m[10]#011train-rmse:3.48445#011validation-rmse:5.27694[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[11]#011train-rmse:3.12242#011validation-rmse:5.00091[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 26 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[12]#011train-rmse:2.83098#011validation-rmse:4.81133[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[13]#011train-rmse:2.62445#011validation-rmse:4.67121[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[14]#011train-rmse:2.44661#011validation-rmse:4.51742[0m
    [31m[15]#011train-rmse:2.30128#011validation-rmse:4.39455[0m
    [31m[16]#011train-rmse:2.19733#011validation-rmse:4.3081[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[17]#011train-rmse:2.09813#011validation-rmse:4.26258[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[18]#011train-rmse:2.01449#011validation-rmse:4.22429[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 22 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[19]#011train-rmse:1.93576#011validation-rmse:4.16988[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[20]#011train-rmse:1.87679#011validation-rmse:4.15776[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[21]#011train-rmse:1.8271#011validation-rmse:4.13839[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[22]#011train-rmse:1.80922#011validation-rmse:4.12826[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[23]#011train-rmse:1.79178#011validation-rmse:4.11281[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[24]#011train-rmse:1.75248#011validation-rmse:4.07431[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[25]#011train-rmse:1.71488#011validation-rmse:4.02159[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[26]#011train-rmse:1.67937#011validation-rmse:4.02045[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 4 pruned nodes, max_depth=5[0m
    [31m[27]#011train-rmse:1.64315#011validation-rmse:4.03391[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 20 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[28]#011train-rmse:1.58817#011validation-rmse:4.02946[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[29]#011train-rmse:1.55179#011validation-rmse:4.01368[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[30]#011train-rmse:1.52515#011validation-rmse:3.97878[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[31]#011train-rmse:1.47182#011validation-rmse:3.98904[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[32]#011train-rmse:1.46473#011validation-rmse:3.97773[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[33]#011train-rmse:1.44261#011validation-rmse:3.96236[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 6 pruned nodes, max_depth=5[0m
    [31m[34]#011train-rmse:1.41975#011validation-rmse:3.95449[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[35]#011train-rmse:1.388#011validation-rmse:3.95119[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 0 pruned nodes, max_depth=5[0m
    [31m[36]#011train-rmse:1.35127#011validation-rmse:3.9715[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 8 pruned nodes, max_depth=5[0m
    [31m[37]#011train-rmse:1.30556#011validation-rmse:3.94535[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 6 pruned nodes, max_depth=5[0m
    [31m[38]#011train-rmse:1.27791#011validation-rmse:3.94688[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 18 extra nodes, 4 pruned nodes, max_depth=5[0m
    [31m[39]#011train-rmse:1.25033#011validation-rmse:3.96629[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 16 extra nodes, 6 pruned nodes, max_depth=5[0m
    [31m[40]#011train-rmse:1.22347#011validation-rmse:3.98427[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 8 pruned nodes, max_depth=5[0m
    [31m[41]#011train-rmse:1.21089#011validation-rmse:3.98901[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 12 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[42]#011train-rmse:1.18709#011validation-rmse:3.98538[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 10 extra nodes, 8 pruned nodes, max_depth=5[0m
    [31m[43]#011train-rmse:1.17602#011validation-rmse:3.9797[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 8 pruned nodes, max_depth=5[0m
    [31m[44]#011train-rmse:1.1385#011validation-rmse:3.97881[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 6 extra nodes, 14 pruned nodes, max_depth=3[0m
    [31m[45]#011train-rmse:1.12919#011validation-rmse:3.97166[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 14 extra nodes, 2 pruned nodes, max_depth=5[0m
    [31m[46]#011train-rmse:1.10531#011validation-rmse:3.97073[0m
    [31m[10:18:35] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 6 extra nodes, 14 pruned nodes, max_depth=3[0m
    [31m[47]#011train-rmse:1.09369#011validation-rmse:3.96122[0m
    [31mStopping. Best iteration:[0m
    [31m[37]#011train-rmse:1.30556#011validation-rmse:3.94535
    [0m
    
    2019-11-30 10:18:47 Completed - Training job completed
    Training seconds: 48
    Billable seconds: 48


### Build the model

Now that the training job has completed, we have some model artifacts which we can use to build a model. Note that here we mean SageMaker's definition of a model, which is a collection of information about a specific algorithm along with the artifacts which result from a training job.


```python
# We begin by asking SageMaker to describe for us the results of the training job. The data structure
# returned contains a lot more information than we currently need, try checking it out yourself in
# more detail.
training_job_info = session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)

model_artifacts = training_job_info['ModelArtifacts']['S3ModelArtifacts']
```


```python
# Just like when we created a training job, the model name must be unique
model_name = training_job_name + "-model"

# We also need to tell SageMaker which container should be used for inference and where it should
# retrieve the model artifacts from. In our case, the xgboost container that we used for training
# can also be used for inference.
primary_container = {
    "Image": container,
    "ModelDataUrl": model_artifacts
}

# And lastly we construct the SageMaker model
model_info = session.sagemaker_client.create_model(
                                ModelName = model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = primary_container)
```

## Step 5: Testing the model

Now that we have fit our model to the training data, using the validation data to avoid overfitting, we can test our model. To do this we will make use of SageMaker's Batch Transform functionality. In other words, we need to set up and execute a batch transform job, similar to the way that we constructed the training job earlier.

### Set up the batch transform job

Just like when we were training our model, we first need to provide some information in the form of a data structure that describes the batch transform job which we wish to execute.

We will only be using some of the options available here but to see some of the additional options please see the SageMaker documentation for [creating a batch transform job](https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html).


```python
# Just like in each of the previous steps, we need to make sure to name our job and the name should be unique.
transform_job_name = 'boston-xgboost-batch-transform-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Now we construct the data structure which will describe the batch transform job.
transform_request = \
{
    "TransformJobName": transform_job_name,
    
    # This is the name of the model that we created earlier.
    "ModelName": model_name,
    
    # This describes how many compute instances should be used at once. If you happen to be doing a very large
    # batch transform job it may be worth running multiple compute instances at once.
    "MaxConcurrentTransforms": 1,
    
    # This says how big each individual request sent to the model should be, at most. One of the things that
    # SageMaker does in the background is to split our data up into chunks so that each chunks stays under
    # this size limit.
    "MaxPayloadInMB": 6,
    
    # Sometimes we may want to send only a single sample to our endpoint at a time, however in this case each of
    # the chunks that we send should contain multiple samples of our input data.
    "BatchStrategy": "MultiRecord",
    
    # This next object describes where the output data should be stored. Some of the more advanced options which
    # we don't cover here also describe how SageMaker should collect output from various batches.
    "TransformOutput": {
        "S3OutputPath": "s3://{}/{}/batch-bransform/".format(session.default_bucket(),prefix)
    },
    
    # Here we describe our input data. Of course, we need to tell SageMaker where on S3 our input data is stored, in
    # addition we need to detail the characteristics of our input data. In particular, since SageMaker may need to
    # split our data up into chunks, it needs to know how the individual samples in our data file appear. In our
    # case each line is its own sample and so we set the split type to 'line'. We also need to tell SageMaker what
    # type of data is being sent, in this case csv, so that it can properly serialize the data.
    "TransformInput": {
        "ContentType": "text/csv",
        "SplitType": "Line",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": test_location,
            }
        }
    },
    
    # And lastly we tell SageMaker what sort of compute instance we would like it to use.
    "TransformResources": {
            "InstanceType": "ml.m4.xlarge",
            "InstanceCount": 1
    }
}
```

### Execute the batch transform job

Now that we have created the request data structure, it is time to ask SageMaker to set up and run our batch transform job. Just like in the previous steps, SageMaker performs these tasks in the background so that if we want to wait for the transform job to terminate (and ensure the job is progressing) we can ask SageMaker to wait of the transform job to complete.


```python
transform_response = session.sagemaker_client.create_transform_job(**transform_request)
```


```python
transform_desc = session.wait_for_transform_job(transform_job_name)
```

    ..........................................!


### Analyze the results

Now that the transform job has completed, the results are stored on S3 as we requested. Since we'd like to do a bit of analysis in the notebook we can use some notebook magic to copy the resulting output from S3 and save it locally.


```python
transform_output = "s3://{}/{}/batch-bransform/".format(session.default_bucket(),prefix)
```


```python
!aws s3 cp --recursive $transform_output $data_dir
```

    download: s3://sagemaker-ap-southeast-1-956989553438/boston-xgboost-LL/batch-bransform/test.csv.out to ../data/boston/test.csv.out


To see how well our model works we can create a simple scatter plot between the predicted and actual values. If the model was completely accurate the resulting scatter plot would look like the line $x=y$. As we can see, our model seems to have done okay but there is room for improvement.


```python
Y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
```


```python
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")
```




    Text(0.5, 1.0, 'Median Price vs Predicted Price')



![png]({{" ../assets/images/output_33_1.png" }})


## Optional: Clean up

The default notebook instance on SageMaker doesn't have a lot of excess disk space available. As you continue to complete and execute notebooks you will eventually fill up this disk space, leading to errors which can be difficult to diagnose. Once you are completely finished using a notebook it is a good idea to remove the files that you created along the way. Of course, you can do this from the terminal or from the notebook hub if you would like. The cell below contains some commands to clean up the created files from within the notebook.


```python
# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir
```


```python

```
