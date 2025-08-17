Getting Started
Prerequisites
Python 3.8+

pip

Docker (optional, for containerized deployment)

Installation
Clone the repository:

git clone <repository-url>
cd resume-screening-engine

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:

pip install -r requirements.txt

Download the spaCy model:

python -m spacy download en_core_web_sm

Training the Model
Before you can run the main application, you need to train the classification model.

python model_builder.py

This script will read the data/resumes.csv file, train a TensorFlow model, and save it to the models/ directory.

Running the Application
To process a single resume and see the output:

python main.py --file resumes/sample_resume_se.txt

You can replace resumes/sample_resume_se.txt with the path to any resume you want to screen.

How It Works
resume_parser.py: This script handles the parsing of resume files.

It reads the text from the file.

It uses spaCy to perform NER to extract entities like names, emails, phone numbers, and skills from the text.

It returns a structured dictionary (JSON) of the extracted information.

model_builder.py: This script is responsible for building and training the ML model.

It loads the sample resume data from data/resumes.csv.

It preprocesses the text data using Tokenizer and padding.

It builds a simple sequential neural network using TensorFlow/Keras.

It trains the model on the data and saves the trained model to the models/ directory.

main.py: This is the main entry point of the application.

It takes a file path as a command-line argument.

It loads the pre-trained TensorFlow model.

It uses resume_parser.py to parse the resume and extract its text.

It uses the loaded model to predict the category of the resume.

It then uses the parser again to extract a structured profile.

Finally, it prints the classification and the structured profile as a JSON object.

Containerization with Docker
You can build and run this project as a Docker container.

Build the Docker image:

docker build -t resume-screening-engine .

Run the Docker container:
To run the screening on a sample resume inside the container:

docker run resume-screening-engine --file resumes/sample_resume_ds.txt

This encapsulates the entire application and its dependencies, making it highly portable.