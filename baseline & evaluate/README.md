AutoGraph
======================================

## Contents
ingestion/: The code and libraries used on Codalab to run your submmission.

scoring/: The code and libraries used on Codalab to score your submmission.

code_submission/: An example of code submission you can use as template.

data/: Some sample data to test your code before you submit it.

run_local_test.py: A python script to simulate the runtime in codalab

`test_i.csv`: The correct results for each dataset.

## Local development and testing
1. To make your own submission to AutoGraph challenge, you need to modify the
file `model.py` in `code_submission/`, which implements your algorithm.
2. Test the algorithm on your local computer using Docker,
in the exact same environment as on the CodaLab challenge platform. Advanced
users can also run local test without Docker, if they install all the required
packages.
3. If you are new to docker, install docker from https://docs.docker.com/get-started/. (Please make sure that you have installed **CUDA, docker and nvidia-docker** before running)
Then, at the shell, run:
```
cd path/to/autograph_starting_kit/
docker run --gpus=0 -it --rm -v "$(pwd):/app/autograph" -w /app/autograph nehzux/kddcup2020:v2
```
The option `-v "$(pwd):/app/autograph"` mounts current directory
(`autograph_starting_kit/`) as `/app/autograph`. If you want to mount other
directories on your disk, please replace `$(pwd)` by your own directory.

The Docker image
```
nehzux/kddcup2020:v2
```

4. You will then be able to run the `ingestion program` (to produce predictions)
and the `scoring program` (to evaluate your predictions) on toy sample data.
In the AutoGraph challenge, both two programs will run in parallel to give
feedback. So we provide a Python script to simulate this behavior. To test locally, run:
```
python run_local_test.py
```
If the program exits without any errors, you can find the final score from the terminal's stdout of your solution.
Also you can view the score by opening the `scoring_output/scores.txt`.

The full usage is
```
python run_local_test.py --dataset_dir=./data/demo --code_dir=./code_submission
```
You can change the argument `dataset_dir` to other datasets On the other hand, you can also modify the directory containing your other sample code.


## Submission

1. You just need to modify `model.py` in `code_submission/`, which implements your algorithm. If you have any other changes, please explain where the other documents have been changed and how to run the project in your report.

2. You need to submit **source code** and **report**.

## Datasets (uploaded to WEB LEARNING before)
### The training data: 
1. `underexpose_item_feat.csv`: the columns of which are: *item_id, txt_vec, img_vec*
2. `underexpose_user_feat.csv`: the columns of which are: u*ser_id, user_age_level, user_gender, user_city_level*
3. `train.csv`: columns are: *user_id, item_id, time*

### The test data:
`test_id.csv`: *user_id, query_time*

### Column:
1. **txt_vec**: the item's text feature, which is a 128-dimensional real-valued vector produced by a pre-trained model
2. **img_vec**: the item's image feature, which is a 128-dimensional real-valued vector produced by a pre-trained model
3. **user_id**: the unique identifier of the user
4. **item_id**: the unique identifier of the item
5. **time**: timestamp when the click event happens, i.e.,（unix_timestamp - random_number_1）/ random_number_2
6. **user_age_level**: the age group to which the user belongs
7. **user_gender**: the gender of the user, which can be empty
8. **user_city_level**: the tier to which the user's city belongs