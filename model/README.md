# Andy's super cool fat cat model

![Surprised Pikachu Face](https://i.kym-cdn.com/entries/icons/original/000/027/475/Screen_Shot_2018-10-25_at_11.02.15_AM.png)

This model classifies a user-given cat photo as either `skinny`, `normal`, or `obese`.

# Installation

Using Conda:

```
conda create --name ObeseCats --file requirements-conda.txt
conda activate ObeseCats
```

Using Pip:

```
python3 -m venv ObeseCats
source ObeseCats/bin/activate
pip install -r requirements-pip.txt
```

# Usage

Run the following after having activated the virtual environment created earlier:

```
python inference.py
```

When prompted, enter the path to the photo and hit `Enter`.

# Additional Usage

Generate the `annotations.csv` file by running `annotate_imgs.py`.

Check out how I am preprocessing the images by running `preprocessing.py`.

Train the model yourself by running `train.py`

Evaluate the model on the test dataset by running `evaluation.py`

