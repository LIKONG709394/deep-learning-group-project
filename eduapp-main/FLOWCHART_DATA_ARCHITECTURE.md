```mermaid
graph LR
    subgraph Input["📥 Input Layer"]
        A1["📄 PDF Upload"]
        A2["📸 Camera Scan"]
        A3["🔗 URL Import"]
    end
    
    subgraph Process["⚙️ Processing Layer"]
        B1["Extract Text<br/>& Metadata"]
        B2["Parse Structure<br/>Sections"]
        B3["Clean & Normalize<br/>Content"]
    end
    
    subgraph AI["🤖 AI Layer<br/>Gemini API"]
        C1["Analyze<br/>Content"]
        C2["Extract<br/>Key Concepts"]
        C3["Generate<br/>Summaries"]
    end
    
    subgraph Generate["✨ Generate Layer"]
        D1["📚 Study Path<br/>Sections"]
        D2["🃏 Flashcards<br/>Q&A Pairs"]
        D3["🎬 Video Scripts<br/>90-sec Videos"]
        D4["❓ Quiz<br/>Questions"]
        D5["🎤 Audio Script<br/>Narration Text"]
    end
    
    subgraph Synthesis["🔊 Synthesis Layer"]
        E1["Text-to-Speech<br/>API"]
        E2["Video Generation<br/>Engine"]
        E3["Audio Processing<br/>Normalize"]
    end
    
    subgraph Storage["💾 Storage Layer"]
        F1["☁️ Cloud Storage<br/>PDFs & Media"]
        F2["🗄️ AsyncStorage<br/>Local Cache"]
        F3["📱 Device Storage<br/>Offline Data"]
    end
    
    subgraph Display["📲 Display Layer"]
        G1["🗺️ Study Path<br/>S-Curve UI"]
        G2["🃏 Flashcard<br/>Flip UI"]
        G3["📹 Reels Feed<br/>Swipeable"]
        G4["🎵 Player<br/>Spotify-Style"]
        G5["📊 Analytics<br/>Dashboard"]
    end
    
    %% Data Flow
    A1 --> B1
    A2 --> B1
    A3 --> B1
    
    B1 --> B2
    B2 --> B3
    B3 --> C1
    
    C1 --> C2
    C2 --> C3
    C3 --> D1
    C3 --> D2
    C3 --> D3
    C3 --> D4
    C3 --> D5
    
    D3 --> E2
    D5 --> E1
    
    E1 --> E3
    E2 --> Synthesis
    E3 --> F1
    
    D1 --> F2
    D2 --> F2
    D4 --> F2
    
    F1 --> F2
    F2 --> F3
    
    F2 --> G1
    F2 --> G2
    F2 --> G3
    F2 --> G4
    F2 --> G5
    
    %% Styling
    classDef input fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#000
    classDef process fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#000
    classDef ai fill:#E8F5E9,stroke:#388E3C,stroke-width:2px,color:#000
    classDef generate fill:#FFF3E0,stroke:#F57C00,stroke-width:2px,color:#000
    classDef synthesis fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#000
    classDef storage fill:#E0F2F1,stroke:#00897B,stroke-width:2px,color:#000
    classDef display fill:#F1F8E9,stroke:#558B2F,stroke-width:2px,color:#000
    
    class A1,A2,A3 input
    class B1,B2,B3 process
    class C1,C2,C3 ai
    class D1,D2,D3,D4,D5 generate
    class E1,E2,E3 synthesis
    class F1,F2,F3 storage
    class G1,G2,G3,G4,G5 display
```

---

## Data Flow Architecture Explained

### 📥 **Input Layer**
- Users upload PDFs, scan with camera, or import from URLs
- Multiple input methods for flexibility

### ⚙️ **Processing Layer**
- Extract raw text and metadata from PDFs
- Parse document structure (chapters, sections, headings)
- Clean and normalize content for AI processing

### 🤖 **AI Layer (Gemini API)**
- Analyze content comprehensively
- Extract key concepts and definitions
- Generate summaries and explanations

### ✨ **Generate Layer**
- **Study Path**: Break content into progressive sections
- **Flashcards**: Create Q&A pairs for spaced repetition
- **Video Scripts**: Generate 90-second summary videos
- **Quiz**: Generate practice questions with answers
- **Audio Script**: Create narration text for TTS

### 🔊 **Synthesis Layer**
- **Text-to-Speech**: Convert audio scripts to natural speech
- **Video Generation**: Create visual content from scripts
- **Audio Processing**: Normalize and optimize audio quality

### 💾 **Storage Layer**
- **Cloud Storage**: Backup PDFs and generated media
- **AsyncStorage**: Local caching for quick access
- **Device Storage**: Offline access to downloaded content

### 📲 **Display Layer**
- **Study Path**: Interactive S-curve visualization
- **Flashcards**: Flip and review cards
- **Reels**: Swipeable short-form videos
- **Player**: Spotify-style audio playback
- **Analytics**: Track progress and statistics
```

Now let me create a third diagram for the app navigation structure:
