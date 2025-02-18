import React, { useState } from "react";

function App() {
  const [csvFile, setCsvFile] = useState(null);
  const [numRows, setNumRows] = useState(50);
  const [epochs, setEpochs] = useState(50);
  const [responseData, setResponseData] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!csvFile) {
      alert("Please select a CSV file first.");
      return;
    }

    try {
      const formData = new FormData();
      formData.append("file", csvFile);
      formData.append("num_rows", numRows);
      formData.append("epochs", epochs);

      const response = await fetch("http://127.0.0.1:8000/generate_synthetic", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }
      
      const csvText = await response.text();
      setResponseData(csvText); // store it for downloading
    } catch (error) {
      console.error("Error:", error);
      alert(`Failed to generate synthetic data: ${error.message}`);
    }
  };

  const handleDownload = () => {
    // if there's no CSV, do nothing
    if (!responseData) return;
    // create a blob and trigger the download
    const blob = new Blob([responseData], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "synthetic_data.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ margin: "20px" }}>
      <h1>Synthetic Data Generator</h1>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: "10px" }}>
          <label htmlFor="csvFile">Upload your CSV File: </label>
          <input
            type="file"
            id="csvFile"
            accept=".csv"
            onChange={(e) => setCsvFile(e.target.files[0])}
          />
        </div>

        <div style={{ marginBottom: "10px" }}>
          <label htmlFor="numRows">Number of rows (CSV output): </label>
          <input
            type="number"
            id="numRows"
            value={numRows}
            onChange={(e) => setNumRows(e.target.value)}
            min="1"
          />
        </div>

        <div style={{ marginBottom: "10px" }}>
          <label htmlFor="epochs">Epochs (Bigger the number, slower the process): </label>
          <input
            type="number"
            id="epochs"
            value={epochs}
            onChange={(e) => setEpochs(e.target.value)}
            min="1"
          />
        </div>

        <button type="submit">Generate Synthetic</button>
      </form>

      {/* Only show a message and download button if we have CSV content */}
      {responseData && (
        <>
          <p>The synthetic data is ready to be downloaded!</p>
          <button onClick={handleDownload}>Download CSV</button>
        </>
      )}
    </div>
  );
}

export default App;
