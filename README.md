# YouTube Transcript + RAG Chatbot

A powerful RAG (Retrieval-Augmented Generation) based YouTube chatbot that extracts transcripts from YouTube videos and answers questions using Groq LLM.

## 🚀 Features

- **YouTube Transcript Extraction**: Automatically extracts captions from YouTube videos using yt-dlp
- **Smart RAG Implementation**: Uses vector embeddings and FAISS for efficient retrieval
- **Groq LLM Integration**: Powered by Groq's fast LLM for accurate answers
- **Cost-Efficient**: Only sends relevant transcript chunks to reduce API costs
- **Multiple Caption Formats**: Supports both manual and auto-generated captions
- **Fallback Support**: Uses Hugging Face model when Groq is unavailable

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kaustubh-d-IITR/YT_CHATBOT.git
   cd YT_CHATBOT
   ```

2. **Create virtual environment**
   ```bash
   python -m venv YT_CHATBOT_env
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   YT_CHATBOT_env\Scripts\activate
   
   # Linux/Mac
   source YT_CHATBOT_env/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 Setup

### Groq API Key

1. Get a free API key from [Groq Console](https://console.groq.com)
2. Set the API key in the app sidebar or as environment variable:
   ```bash
   set GROQ_API_KEY=your_api_key_here
   ```

## 🎯 Usage

1. **Run the application**
   ```bash
   streamlit run app.py
   ```

2. **Open in browser**
   - Local URL: `http://localhost:8501`
   - Network URL: `http://your-ip:8501`

3. **Enter YouTube URL**
   - Paste any YouTube video URL
   - Click "Get Transcript"

4. **Ask Questions**
   - Enter your question about the video
   - Get AI-powered answers based on transcript content

## 📁 Project Structure

```
YT_CHATBOT/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore rules
└── YT_CHATBOT_env/      # Virtual environment (not tracked)
```

## 🔧 How It Works

### 1. Transcript Extraction
- Uses `yt-dlp` to extract video metadata
- Downloads captions (manual or auto-generated)
- Parses JSON and SRT caption formats

### 2. RAG Implementation
- Splits transcript into 500-word chunks
- Creates embeddings using `all-MiniLM-L6-v2`
- Builds FAISS vector index for similarity search
- Retrieves top 3 most relevant chunks per question

### 3. Question Answering
- Embeds user question
- Finds most similar transcript chunks
- Sends only relevant chunks to Groq LLM
- Returns accurate, context-aware answers

## 💰 Cost Efficiency

- **90%+ reduction** in tokens sent to Groq
- **Smart chunking** prevents context loss
- **Vector search** ensures relevant content
- **Transparency** shows which chunks were used

## 🎬 Supported Videos

- Videos with manual captions (best quality)
- Videos with auto-generated captions
- Multiple language support (English primary)
- Public and unlisted videos

## 🛡️ Error Handling

- Graceful fallback to Hugging Face model
- Clear error messages for troubleshooting
- Multiple caption format support
- Robust URL validation

## 📊 Dependencies

- `streamlit` - Web application framework
- `yt-dlp` - YouTube video processing
- `groq` - LLM API integration
- `sentence-transformers` - Text embeddings
- `faiss-cpu` - Vector similarity search
- `transformers` - Hugging Face models
- `requests` - HTTP requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Groq](https://groq.com) for fast LLM API
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube processing
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section in the app
- Ensure your Groq API key is valid

---

**Made with ❤️ for efficient YouTube content analysis**
