# Project Name

This project is a Python-based image processing tool that uses OpenCV, Tesseract OCR, and Pyzbar to analyze images. It performs OCR to detect numbers, identifies QR codes, and detects colored circles inside a blue rectangle in the image.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Introduction

This project demonstrates how to use Python libraries to process images for various purposes. The script performs the following tasks:

1. Detects numbers in the image using Tesseract OCR.
2. Identifies and decodes QR codes using the Pyzbar library.
3. Detects a blue rectangle in the image using color filtering techniques.
4. Identifies circles inside the detected blue rectangle and determines their colors.
5. Aligns detected numbers with the circles based on their x-axis position.

## Features

- **OCR Number Detection**: Uses Tesseract OCR to extract numbers from the image.
- **QR Code Detection**: Detects and decodes QR codes present in the image.
- **Blue Rectangle Detection**: Identifies a blue rectangle using HSV color space filtering.
- **Circle and Color Detection**: Finds circles within the blue rectangle and detects their color.
- **Number Alignment**: Aligns detected numbers with circles based on their x-axis position.

## Installation

To set up and run this project, follow these steps:

### Prerequisites

- **Python 3.x**: Ensure you have Python 3 installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

### Steps to Install

1. **Clone the Repository**:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment**:

    ```bash
    python3 -m venv venv
    ```

3. **Activate the virtual environment**:

    - On macOS and Linux:

      ```bash
      source venv/bin/activate
      ```

    - On Windows:

      ```bash
      .\venv\Scripts\activate
      ```

4. **Install the required dependencies**:

    Make sure you have a `requirements.txt` file in the project directory, then run:

    ```bash
    pip install -r requirements.txt
    ```

5. **Install Tesseract OCR**:

    - **Ubuntu**: Run `sudo apt-get install tesseract-ocr`.
    - **macOS**: Run `brew install tesseract` (requires Homebrew).
    - **Windows**: Download and install Tesseract from [the official GitHub repository](https://github.com/tesseract-ocr/tesseract). Make sure Tesseract is added to your system's PATH.

## Usage

After setting up the environment and installing the dependencies, you can run the demo script. Ensure you replace `'path/your/images'` in the script with the actual path to your image file.

```bash
python3 demo.py
