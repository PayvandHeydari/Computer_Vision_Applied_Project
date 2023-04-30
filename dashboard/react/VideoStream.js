import React from 'react';

const VideoStream = ({ videoURL }) => {
  return (
    <div>
      <h2>Video Stream</h2>
      <video
        controls
        width="640"
        height="480"
        src={videoURL}
        type="video/mp4"
      >
        Your browser does not support the video tag.
      </video>
    </div>
  );
};

export default VideoStream;
