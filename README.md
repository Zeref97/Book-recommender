# Book Recommender

These recommendation engines were built on the [Goodreads](https://www.goodreads.com/) API,
the [LightFM recommender model](https://github.com/lyst/lightfm) by Maciej Kula
and the [goodbooks-10k book dataset](https://github.com/zygmuntz/goodbooks-10k) by Zygmunt Zając.

## Installation

Once you have cloned the repo
- Create a python virtual environment in whatever way you prefer
- Activate the virtual environment
- [Download model](https://drive.google.com/file/d/1jY58VcsbyOSB572vKZ4Weag5_sOmpK39/view?usp=sharing) and put in to model folder. 
- Run `pip install Cython`
- cd into the top directory of the repo and run `pip install -r requirements.txt`
This should install all of the dependencies in your virtual environment, including the 'lightfm_ext' package.

## Running

To run, cd into the top level of the repository and run `python app.py`. This will launch the Flask app, and will allow you to play around with the recommenders.


## Result

[Video demo](https://youtu.be/r6ARQ4oTzhw)
