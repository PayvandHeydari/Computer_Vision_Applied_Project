import React from 'react';

const ObjectDetectionData = ({ objectData }) => {
  return (
    <div>
      <h2>Object Detection Data</h2>
      <ul>
        {objectData.map((item) => (
          <li key={item.object}>
            {item.object}: {item.count}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ObjectDetectionData;
