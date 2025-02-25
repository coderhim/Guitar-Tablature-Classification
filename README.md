# 🎸 Guitar Tab Classification Project

Welcome to the **Guitar Tab Classification** project! This repository focuses on extracting musical notes from guitar performances using advanced deep learning techniques. We aim to provide an efficient and accurate method for guitar transcription using less expensive finger position estimation and JAMS annotations.

## 📂 Dataset
We use the **GuitarSet** dataset, a comprehensive collection of guitar recordings with detailed annotations.

- **Dataset Link**: [GuitarSet on Zenodo](https://zenodo.org/records/3371780)  
- **Reference**:  
  Q. Xi, R. Bittner, J. Pauwels, X. Ye, and J. P. Bello, "GuitarSet: A Dataset for Guitar Transcription", in 19th International Society for Music Information Retrieval Conference, Paris, France, Sept. 2018.

## 🧠 Methodology
Our project is inspired by the article **"Automated Guitar Transcription with Deep Learning"** and focuses on:

1. **Notes Extraction**: Parsing JAMS annotations to extract guitar notes.  
2. **Finger Position Estimation**: Implementing a computationally efficient approach for estimating finger positions.  
3. **Deep Learning Models**: Leveraging modern deep learning models for automated guitar transcription.

## 📊 Features
- Efficient parsing of JAMS annotations.  
- Improved transcription accuracy using finger position estimation.  
- Supports multiple guitar styles and techniques.  

## 🛠️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/seminar_audioTab.git
   cd seminar_audioTab
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download the GuitarSet dataset and place it in the `dataset` folder.

4. Run the main script:
   ```bash
   python engine.py
   ```

## 🤝 Acknowledgements
Special thanks to the authors of **"GuitarSet"** and the Medium article **"Automated Guitar Transcription with Deep Learning"** for their valuable contributions to the field.

## 📬 Contact
For any questions or contributions, feel free to open an issue or reach out!

---

⭐ **If you find this project useful, consider giving it a star!**

