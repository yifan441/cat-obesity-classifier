# Andy's super cool fat cat model

![Surprised Pikachu Face](https://www.google.com/url?sa=i&url=https%3A%2F%2Fknowyourmeme.com%2Fmemes%2Fsurprised-pikachu&psig=AOvVaw1yFqywPRki5RYI173qbQ3y&ust=1725402948819000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCOjbzoippYgDFQAAAAAdAAAAABAI)

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

