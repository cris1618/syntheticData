import React, { useState, useRef } from "react";

function App() {
  const [csvFile, setCsvFile] = useState(null);
  const [fileUploaded, setFileUploaded] = useState(false); // For "File Uploaded!" message
  const [numRows, setNumRows] = useState(50);
  const [epochs, setEpochs] = useState(50);
  const [responseData, setResponseData] = useState("");
  
  // For our mock progress bar
  const [progress, setProgress] = useState(0);
  const [isTraining, setIsTraining] = useState(false);

  // Reference to hidden file input
  const hiddenFileInput = useRef(null);

  // Trigger hidden input
  const handleSelectFileClick = () => {
    hiddenFileInput.current.click();
  };

  // On file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setCsvFile(file);
      setFileUploaded(true);  // Show "File Uploaded!" message
    } else {
      setFileUploaded(false);
    }
  };

  // This simulates incrementing progress from 0 to 95%
  // Then once the request finishes, we jump to 100%.
  const startProgressSimulation = () => {
    setIsTraining(true);
    setProgress(0);

    // We'll increment progress by 1% every 100ms until 95
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev < 95) {
          return prev + 1;
        } else {
          clearInterval(interval);
          return prev; // stay at 95
        }
      });
    }, 100);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!csvFile) {
      alert("Please select a CSV file first.");
      return;
    }

    // Start "training" progress
    startProgressSimulation();

    try {
      const formData = new FormData();
      formData.append("file", csvFile);
      formData.append("num_rows", parseInt(numRows, 10));
      formData.append("epochs", parseInt(epochs, 10));

      const response = await fetch("http://127.0.0.1:8000/generate_synthetic", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }
      
      const csvText = await response.text();
      setResponseData(csvText);
      
      // Jump to 100% once the fetch completes
      setProgress(100);
    } catch (error) {
      console.error("Error:", error);
      alert(`Failed to generate synthetic data: ${error.message}`);
    } finally {
      // Stop "training" mode
      setTimeout(() => {
        setIsTraining(false);
      }, 800); // small delay so user sees 100% briefly
    }
  };

  const handleDownload = () => {
    if (!responseData) return;
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
    <div style={styles.page}>
      <div style={styles.overlay}></div>
      <h1 style={styles.title}>Synthetic Data Generator</h1>
      
      <h2 style={styles.subtitle}>
        Upload a CSV file and generate a new dataset <br />
        with the number of rows that you choose!
      </h2>

      <form onSubmit={handleSubmit} style={styles.form}>
        {/* Big button to select file */}
        <button
          type="button"
          style={styles.uploadBtn}
          onClick={handleSelectFileClick}
        >
          Select CSV File
        </button>

        {/* Hidden file input */}
        <input
          type="file"
          ref={hiddenFileInput}
          accept=".csv"
          style={{ display: "none" }}
          onChange={handleFileChange}
        />

        {/* Show "File Uploaded!" message if file is chosen */}
        {fileUploaded && (
          <p style={{ marginTop: 5, fontStyle: "italic", color: "#28a745" }}>
            File Uploaded!
          </p>
        )}

        {/* Number of rows */}
        <div style={styles.field}>
          <label htmlFor="numRows">Number of rows (CSV output):</label>
          <input
            type="number"
            id="numRows"
            value={numRows}
            onChange={(e) => setNumRows(e.target.value)}
            min="1"
            style={styles.input}
          />
        </div>

        {/* Epochs */}
        <div style={styles.field}>
          <label htmlFor="epochs">Epochs (Bigger # = slower):</label>
          <input
            type="number"
            id="epochs"
            value={epochs}
            onChange={(e) => setEpochs(e.target.value)}
            min="1"
            style={styles.input}
          />
        </div>

        <button type="submit" style={styles.generateBtn}>
          Generate Synthetic
        </button>

        {/* Progress bar (only show if "training" or if progress < 100) */}
        {isTraining && (
          <div style={styles.progressContainer}>
            <div style={{
              ...styles.progressBar,
              width: `${progress}%`,
            }}>
              {progress}%
            </div>
          </div>
        )}
      </form>

      {/* If we have a CSV, show "Download CSV" button */}
      {responseData && !isTraining && (
        <div style={{ marginTop: "10px", textAlign: "center" }}>
          <p>The synthetic data is ready to be downloaded! <br />
          E quindi uscimmo a riveder le stelle.
          </p>
          <button onClick={handleDownload} style={styles.downloadBtn}>
            Download CSV
          </button>
        </div>
      )}
    </div>
  );
}

const styles = {
  //page: {
    //minHeight: "100vh",
    //display: "flex",
    //flexDirection: "column",
    //alignItems: "center",
    //background: "#f5f5f5",
    //padding: "40px",
  //},
  page: {
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",  // Needed for overlay
    backgroundImage: "linear-gradient(rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.9)), url('/thelovers.jpg')",
    backgroundSize: "cover",
    backgroundPosition: "center",
    backgroundRepeat: "no-repeat",
  },
  overlay: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    backgroundColor: "rgba(255, 255, 255, 0.5)", // Black with 50% opacity
    zIndex: -1,  // Places it behind the content
  },
  title: {
    marginBottom: "10px",
    fontSize: "1.8rem",
    textAlign: "center",
  },
  subtitle: {
    marginTop: 0,
    marginBottom: "20px",
    textAlign: "center",
    fontSize: "1.2rem"
  },
  form: {
    background: "#fff",
    borderRadius: "10px",
    boxShadow: "0 0 10px rgba(0,0,0,0.1)",
    padding: "20px",
    width: "300px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },
  uploadBtn: {
    backgroundColor: "#222222",
    color: "#fff",
    border: "none",
    padding: "15px 30px",
    fontSize: "1rem",
    borderRadius: "5px",
    cursor: "pointer",
    marginBottom: "10px",
  },
  field: {
    marginBottom: "15px",
    width: "100%",
    display: "flex",
    flexDirection: "column",
  },
  input: {
    marginTop: "5px",
    padding: "8px",
    fontSize: "1rem",
  },
  generateBtn: {
    backgroundColor: "#222222",  // #28a745
    color: "#fff",
    border: "none",
    padding: "15px 30px",
    fontSize: "1rem",
    borderRadius: "5px",
    cursor: "pointer",
    marginTop: "10px",
    marginBottom: "10px",
  },
  progressContainer: {
    width: "100%",
    backgroundColor: "#e0e0e0",
    borderRadius: "5px",
    overflow: "hidden",
    marginTop: "10px",
  },
  progressBar: {
    height: "20px",
    backgroundColor: "#007BFF",
    color: "#fff",
    textAlign: "center",
    lineHeight: "20px",
    transition: "width 0.3s ease",
    fontSize: "0.8rem",
  },
  downloadBtn: {
    backgroundColor: "#222222", // #ff5722
    color: "#fff",
    border: "none",
    padding: "15px 30px",
    fontSize: "1rem",
    borderRadius: "5px",
    cursor: "pointer",
  },
};

export default App;
