import { useState, useEffect, useRef } from 'react'

function App() {
  return (
    <div className='h-screen w-screen flex items-center justify-center'>
      <PointEditor />
    </div>
  );
}

function PointEditor() {
  const [points, setPoints] = useState([]);
  const [labels, setLabels] = useState([]);
  const [exclude, setExclude] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [imageURL, setImageURL] = useState(null);
  const [imgScale, setImgScale] = useState(null);
  const [responseURL, setResponseURL] = useState(null);

  const imgRef = useRef(null);

  useEffect(() => {
    console.log(`Points: ${points}, Labels: ${labels}`);
  }, [points, labels]);

  useEffect(() => {
    if(!imageURL) {
      return;
    }

    return () => {
      URL.revokeObjectURL(imageURL);
    }
  }, [imageURL]);

  useEffect(() => {
    if(imgScale) {
      // console.log(`img scale: ${imgScale.x}, ${imgScale.y}`);
    }
  }, [imgScale]);

  const getImgScale = () => {
    if(imgRef.current) {
      const rect = imgRef.current.getBoundingClientRect();
      const xScale = imgRef.current.naturalWidth / rect.width;
      const yScale = imgRef.current.naturalHeight / rect.height;
      setImgScale({x: xScale, y: yScale});
    }
  };

  const addPoint = (event) => {
    if(imageURL) {
      const rect = event.currentTarget.getBoundingClientRect();
      const x = Math.round(event.clientX - rect.left);
      const y = Math.round(event.clientY - rect.top);
      setPoints(prev => [...prev, [x, y]]);
      setLabels(prev => [...prev, exclude ? 0 : 1]);
    }
  };

  const undo = () => {
    setPoints(points.slice(0, -1));
    setLabels(labels.slice(0, -1));
  };

  const showImage = (e) => {
    const file = e.target.files[0];

    if(file) {
      setImageFile(file);
      setImageURL(URL.createObjectURL(file));
    }
    
  };

  const sendImage = async () => {
    if(points.length > 0 && imageURL) {
      const formData = new FormData();
      formData.append("file", imageFile);
      formData.append("points", JSON.stringify(
        points.map((pnt) => (
          [Math.round(pnt[0] * imgScale.x), Math.round(pnt[1] * imgScale.y)]
        ))));
      formData.append("labels", JSON.stringify(labels));

      const response = await fetch("http://0.0.0.0:8000/segment", {
        method: "POST",
        body: formData
      });

      // console.log(`response: ${response.}`);
      console.log("recieved");
      if(response.ok) {
        console.log("response ok");
        const blob = await response.blob();
        setResponseURL(URL.createObjectURL(blob));
      } else {
        console.log(`not ok, status code: ${response.status}`);
      }
    }

    // if(!response.ok) {
    //   setText("did not work");
    // }

    // const blob = await response.blob();
    // setSegmentedURL(URL.createObjectURL(blob));
  };

  let bgColor = exclude ? 'bg-stone-400' : 'bg-stone-300';

  return (
    <div className='h-[800px] w-[1000px] border-2 border-slate-600 flex flex-row'>
      <div className='w-4/5 flex items-center'>
        <div className='max-w-full max-h-full relative' onClick={addPoint}>
          <img
            src={imageURL}
            className="max-w-full max-h-full"
            ref={imgRef}
            onLoad={getImgScale}
          />
          {points.map((pnt, index) => (
            <Dot index={index} pnt={pnt} labels={labels}/>
          ))}
        </div>
      </div>
      <div className='w-1/5 bg-gray-200 border-l-2 border-slate-600 flex flex-col items-center gap-[10px]'>
        <div className='relative h-20 w-full'>

        </div>
        <button
          onClick={undo}
          className='w-2/3'
        >
            Undo
        </button>
        <div
          className={`h-8 w-2/3 rounded-lg ${bgColor} flex justify-center items-center border border-transparent hover:border-slate-600`}
          onClick={() => setExclude(!exclude)}
        >
          <p>
            Exclude
          </p>
        </div>
        <div className='h-full w-full flex flex-col-reverse'>
          <div className='h-1/3 w-full bg-blue-200 self-center flex flex-col items-center justify-center gap-10'>
            {/* <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              className="mb-4"
            /> */}
            <input
              type="file"
              accept="image/*"
              onChange={showImage}
              className="w-full"
            />
            <button
              onClick={sendImage}
              className='w-2/3'
            >
              Send Image
            </button>
          </div>
          {responseURL && <img src={responseURL} className='max-w-full' />}
        </div>
      </div>
    </div>
  );
  
}

function Dot({index, pnt, labels}) {
  let color = labels[index] == 0 ? 'bg-red-600' : 'bg-indigo-600';

  return (
    <div
      key={index}
      className={`absolute w-3 h-3 rounded-full ${color}`}
      style={{ left: pnt[0], top: pnt[1], transform: 'translate(-50%, -50%)' }}
    />
  )
}

function Canvas() {
  
}

// function SelectButton({ text, onPress }) {
//   return (
//     <div className={``}>

//     </div>
//   );
// }

export default App;
