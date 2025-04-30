import { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [originalImage, setOriginalImage] = useState(null)
  const [originalImageURL, setOriginalImageURL] = useState(null);
  const [text, setText] = useState("test");
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [clickedPos, setClickedPos] = useState({ x: 0, y: 0 });
  const [segmentedURL, setSegmentedURL] = useState(null)

  // useEffect(() => {
  //   const handleMouseMove = (event) => {
  //     setMousePos({ x: event.clientX, y: event.clientY });
  //   };
  //   window.addEventListener("mousemove", handleMouseMove);
  //   return () => {
  //     window.removeEventListener(mousemove, handleMouseMove);
  //   };
  // }, []);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setOriginalImage(file);
    setOriginalImageURL(URL.createObjectURL(file));
  };

  const testEndpoint = async () => {
    const response = await fetch("http://0.0.0.0:8000/");
    console.log("got response");
    console.log((await response.json()).message);
  };

  const sendImage = async () => {
    const formData = new FormData();
    formData.append("file", originalImage);
    formData.append("points", JSON.stringify([[clickedPos.x, clickedPos.y]]))
    formData.append("labels", JSON.stringify([1]))
    const response = await fetch("http://0.0.0.0:8000/segment", {
      method: "POST",
      mode: 'no-cors',
      body: formData
    });

    if(!response.ok) {
      setText("did not work");
    }

    const blob = await response.blob();
    setSegmentedURL(URL.createObjectURL(blob));
  };

  // const handleMouseMove = (event) => {
  //   const rect = event.currentTarget.getBoundingClientRect();
  //   console.log(rect)
  //   setMousePos({ x: Math.round(event.clientX - rect.left), y: Math.round(event.clientY - rect.top) });
  // };

  const handleMouseClick = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    console.log(rect)
    setClickedPos({ x: Math.round(event.clientX - rect.left), y: Math.round(event.clientY - rect.top) });
  }

  console.log("dfsf");
  return (
    <div className="p-4 border w-full">
      <input
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        className="mb-4"
      />

      {originalImageURL && (
        <div>
          <p className="mb-2">Image Preview:</p>
          <img
            src={originalImageURL}
            alt="preview"
            className="max-w-xs border rounded"
            onMouseDown={handleMouseClick}
          />
        </div>
      )}
      <button onClick={sendImage}>Send Image</button>
      {/* <h1>{text}</h1>
      <div
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseClick}
        className='outline'
      >
        <h1>Mouse Position: ({mousePos.x}, {mousePos.y})</h1>
      </div> */}
      <h4>Clicked Pos: ({clickedPos.x}, {clickedPos.y})</h4>
      {segmentedURL && (
        <div>
          <p className='mb-2'>Segmented Image:</p>
          <img
            src={segmentedURL}
            alt="segmented"
            className="max-w-xs border rounded"
          />
        </div>
      )}
    </div>
  );
}

export default App;
