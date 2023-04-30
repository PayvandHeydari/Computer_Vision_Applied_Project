import React, { useState, useEffect } from 'react';
import axios from 'axios';
import HardwareSpecs from './components/HardwareSpecs';
import ObjectDetectionData from './components/ObjectDetectionData';
import VideoStream from './components/VideoStream';
import './App.css';

function App() {
  const [hardwareData, setHardwareData] = useState({});
  const [objectData, setObjectData] = useState([]);
  const videoURL = "http://localhost:5000/api/static/videos/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.mp4";

  useEffect(() => {
    fetchHardwareData();
    fetchObjectData();
  }, []);

  const fetchHardwareData = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/hardware');
      setHardwareData(response.data);
    } catch (error) {
      console.error('Error fetching hardware data:', error);
    }
  };

  const fetchObjectData = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/object-data');
      setObjectData(response.data);
    } catch (error) {
      console.error('Error fetching object data:', error);
    }
  };

  return (
    <div className="App">
      <h1>Dashboard</h1>
      <HardwareSpecs hardwareData={hardwareData} />
      <ObjectDetectionData objectData={objectData} />
      <VideoStream videoURL={videoURL} />
      {/* Add more components as needed */}
    </div>
  )
}
export default App;
