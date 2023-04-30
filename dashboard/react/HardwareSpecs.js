import React from 'react';

const HardwareSpecs = ({ hardwareData }) => {
  return (
    <div>
      <h2>Hardware Specs</h2>
      <ul>
        {Object.entries(hardwareData).map(([key, value]) => (
          <li key={key}>
            {key}: {value}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default HardwareSpecs;
