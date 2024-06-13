# Youtube Link to Flashcards Generator

## Project Overview

This project allows users to generate flashcards from Youtube videos. By inputting a Youtube link, the application analyzes the video content and extracts key concepts to create flashcards. This can be especially useful for students and educators looking to summarize educational content quickly.

## Technologies Used

- **Backend**: FastAPI, Python
- **Frontend**: React
- **API Interaction**: Axios
- **AI Services**: Custom AI Processors (`YoutubeProcessor`, `GeminiProcessor`)

## Project Structure

### Backend

- **File**: `main.py``main2.py`
- **Framework**: FastAPI
- **Endpoints**:
  - `POST /analyze_video`: Analyzes the Youtube video and extracts key concepts.
  - `GET /root`: Health check endpoint.
- **Middlewares**: CORS (Cross-Origin Resource Sharing)

### Frontend

- **File**: `App.jsx``App2.jsx`
- **Framework**: React
- **Components**:
  - `App`: Main application component handling the Youtube link input and displaying flashcards.
  - `Flashcard`: Individual flashcard component for displaying key concepts.

## Installation and Setup

### Backend

1. **Install Dependencies**:
    ```sh
    pip install fastapi uvicorn pydantic
    ```

2. **Run the Server**:
    ```sh
    uvicorn main2:app --reload
    ```
    This will start the FastAPI server on `http://localhost:8000`.

### Frontend

1. **Install Dependencies**:
    ```sh
    npm install
    ```

2. **Run the React Application**:
    ```sh
    npm start
    ```
    This will start the React development server on `http://localhost:3000`.

## Usage

1. **Start the Backend Server**:
    Ensure the FastAPI server is running on `http://localhost:8000`.

2. **Start the Frontend Application**:
    Ensure the React application is running on `http://localhost:3000`.

3. **Generate Flashcards**:
    - Open the React application in your browser.
    - Paste a Youtube link into the input field.
    - Click the "Generate Flashcards" button.
    - The application will call the backend API to analyze the video and display the generated flashcards.

## Detailed Component Descriptions

### Backend: `main2.py`

- **VideoAnalysisRequest**: Defines the request model for analyzing Youtube videos.
- **CORS Middleware**: Configured to allow all origins, methods, and headers.
- **`/analyze_video` Endpoint**: Handles the analysis of the provided Youtube link by calling `YoutubeProcessor` and `GeminiProcessor` to retrieve and process video content.
- **`/root` Endpoint**: Simple health check endpoint to ensure the server is running.

### Frontend: `App2.jsx`

- **State Management**: Uses React hooks (`useState`) to manage the Youtube link input and key concepts state.
- **API Call**: Uses Axios to send the Youtube link to the backend and retrieve key concepts.
- **Flashcard Display**: Maps over the key concepts to display them as individual flashcards using the `Flashcard` component.
- **Discard Flashcard**: Allows users to discard individual flashcards.

### Flashcard Component: `Flashcard.jsx`

- **Props**: Accepts `term`, `definition`, and `onDiscard` as props.
- **Rendering**: Displays the term and definition, and includes a button to discard the flashcard.
