# image-classifier

## Overview

This is a Flask-based web application that classifies uploaded images into categories using a deep learning model. The project allows users to upload an image, and the model predicts whether the image represents a "Happy" or "Sad" emotion.

## Features

- 📷 Upload images via file selection or drag-and-drop.
- 🖥️ Web-based interface built with HTML, CSS, and JavaScript.
- 🧠 Deep learning model for image classification (TensorFlow/Keras).
- 🔥 Real-time image preview before submitting.
- 🎯 Displays predictions instantly after uploading an image.

## Tech Stack

- **Backend:** Flask, TensorFlow/Keras, OpenCV, NumPy
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Local server (Flask)

## Installation

### Prerequisites

Ensure you have **Python 3.7+** installed on your system.

### Step 1: Clone the Repository

```bash
 git clone https://github.com/your-username/image-classifier.git
 cd image-classifier
```

### Step 2: Install Dependencies

```bash
pip install pandas numpy tensorflow flask opencv-python
```

### Step 3: Run the Flask Application

```bash
python app.py
```

The application will be accessible at: [**http://127.0.0.1:5000/**](http://127.0.0.1:5000/)

## Project Structure

```
image-classifier/
│── static/
│   ├── uploads/        # Stores uploaded images
│── templates/
│   ├── index.html      # Frontend UI
│── app.py              # Main Flask application
│── imageclassifier.h5  # Pre-trained model
│── README.md           # Project documentation
```

## How It Works

1. **User uploads an image** through the web interface.
2. **Flask processes the image** and passes it to the deep learning model.
3. **Model predicts** whether the image represents a Happy or Sad emotion.
4. **Result is displayed** on the webpage along with the uploaded image.



## Contributing

Feel free to fork this repository and make improvements. Pull requests are welcome! 🚀


## Contact

For any questions or suggestions, reach out to me at [**seetaram.22jics983@jietjodhpur.ac.in**](mailto\:seetaram.22jics083@jietjodhpur.ac.in) or connect on [LinkedIn](https://www.linkedin.com/in/seetaram-prajapat).

