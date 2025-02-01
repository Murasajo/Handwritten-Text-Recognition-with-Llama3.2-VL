# 🖋️ Handwritten Text Recognition with Llama3.2-VL

This project uses a fine-tuned version of the **Llama3.2-VL** model to extract handwritten text from images. The model is optimized for precision and clarity, making it ideal for digitizing handwritten notes, letters, or documents. The project includes a user-friendly Gradio interface for easy interaction.

---

## 🚀 Features

- **📄 Handwritten Text Extraction**: Accurately extracts handwritten text from images.
- **🛠️ Customizable Prompts**: Users can provide custom prompts to guide the text extraction process.
- **🖥️ User-Friendly Interface**: Built with Gradio for easy interaction.
- **🧠 Fine-Tuned Model**: Uses a LoRA fine-tuned version of the Llama3.2-VL model for improved performance.

---

## 🛠️ Installation

### 💻 System Requirements
- **GPU Requirement**: The model requires **24GB or more of GPU memory** to run efficiently. If you do not have a high-memory GPU, you can use **Google Colab** with a T4 GPU.

### 🚀 Running on Google Colab
If you do not have a local GPU with 24GB+ VRAM, you can use **Google Colab**:
1. Open the `Test Model` notebook in the repository.
2. Select **Runtime > Change runtime type > GPU**.
3. Ensure you are using a **T4 GPU**.
4. Run the cells to set up the environment and test the model.

### 🖥️ Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/handwritten-text-recognition.git
   cd handwritten-text-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Gradio app:
   ```bash
   python app.py
   ```

4. Open the provided link in your browser to access the interface.

---

## 🎯 Usage

1. **📤 Upload an Image**: Use the "Upload Handwritten Text Image" button to upload an image containing handwritten text.
2. **📝 Customize the Prompt (Optional)**: Enter a custom prompt in the textbox or leave it empty to use the default prompt.
3. **🔍 Extract Text**: Click the "Extract Text" button to process the image and extract the handwritten text.
4. **📄 View Results**: The extracted text will appear in the output box.

---

## 📋 Example Prompts

- **Default Prompt**: "Extract all the handwritten text in the image, ensure precision and clarity. The text is handwritten and in English. Display it in a paragraph form as it is written in the image."
- **Custom Prompt**: "Extract only the names and dates from the handwritten text."

---

## 📦 Requirements

- Python 3.8+
- Gradio
- Transformers
- PyTorch
- Pillow
- NumPy
- Hugging Face Hub
- PEFT
- Unsloth
- Bitsandbytes

---

## 🧑‍💻 Code Overview

The project consists of the following components:

- **🧠 Model Loading**: The base Llama3.2-VL model is loaded with 4-bit quantization for efficient memory usage. A LoRA fine-tuned adapter is applied for improved performance.
- **🖼️ Image Processing**: The `process_image` function handles image preprocessing and text generation using the fine-tuned model.
- **🖥️ Gradio Interface**: A user-friendly interface is built using Gradio, allowing users to upload images, customize prompts, and view extracted text.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library and model hosting.
- [Unsloth](https://github.com/unslothai/unsloth) for efficient model fine-tuning.
- [Gradio](https://gradio.app/) for the user-friendly interface.

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the project maintainer.

---
