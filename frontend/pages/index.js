import Head from 'next/head';
import styles from '../styles/Home.module.css';
import { useRef, useEffect, useState } from 'react';

export default function Home() {
  const videoRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    return () => {
      stopStream();
    };
  }, []);

  const startStream = async () => {
    try {
      const response = await fetch('http://localhost:8000/stream');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      videoRef.current.src = url;
      videoRef.current.play();
      setIsStreaming(true);
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const stopStream = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.src = '';
      setIsStreaming(false);
    }
  };

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/trace_video', {
        method: 'POST',
        body: formData,
        mode: 'cors',
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        // Open the traced video in a new window or display it as needed
        window.open(url, '_blank');
      } else {
        console.error('Error uploading file:', response.statusText);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.logoContainer}>
        <img src="/logo.svg" alt="Logo" className={styles.logoImage} />
      </div>
      <div className={styles.header}>
        <a href="/about" className={styles.aboutLink}>
          About us
        </a>
      </div>
      <Head>
        <title>Demo Page</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main>
        <h1 className={styles.title}>SafeSteps</h1>
        <h2 className={styles.h1}>Where every second matters</h2>
        <div className={styles.grid}>
          <div className={styles.card}>
            <h3>Upload File</h3>
            <p>Click the button below to upload a video file:</p>
            <label htmlFor="fileInput" className={styles.button}>
              Choose File
              <input
                id="fileInput"
                type="file"
                onChange={handleUpload}
                style={{ display: 'none' }}
              />
            </label>
          </div>
          <div className={styles.card}>
            <h3>Live Camera Stream</h3>
            <p>
              Click the "Start Stream" button to start the live camera stream.
              Click "Stop Stream" to stop the stream.
            </p>
            <video ref={videoRef} className={styles.video} />
            {!isStreaming && (
              <button onClick={startStream} className={styles.button}>
                Start Stream
              </button>
            )}
            {isStreaming && (
              <button onClick={stopStream} className={styles.button}>
                Stop Stream
              </button>
            )}
          </div>
        </div>
      </main>
      <footer>
        <a
          href="https://vercel.com?utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          Powered by{' '}
          <img src="/vercel.svg" alt="Vercel" className={styles.logo} />
        </a>
      </footer>
    </div>
  );
}