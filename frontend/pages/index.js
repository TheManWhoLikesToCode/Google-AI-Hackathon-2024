import Head from 'next/head';
import styles from '../styles/Home.module.css';
import { useRef, useEffect, useState } from 'react';

export default function Home() {
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    return () => {
      stopStream();
    };
  }, []);
  const startStream = async () => {
    try {
      console.log('Starting stream...');
      const params = {
        camera: 0,
        detection_input_size: 384,
        pose_input_size: '224x160',
        device: 'cpu',
        show_detected: true,
        show_skeleton: true,
      };
      const url = 'http://localhost:8000/stream';
      const queryString = Object.entries(params)
        .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
        .join('&');
      const response = await fetch(`${url}?${queryString}`, { mode: 'cors' });
      console.log('Response status code:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      console.log('Creating a window to display the video stream...');
      const canvas = canvasRef.current;
      if (!canvas) {
        throw new Error('Canvas element not found.');
      }
      setIsStreaming(true);
      const ctx = canvas.getContext('2d');
      let jpg = new Uint8Array();
      let frameCount = 0;
      const reader = response.body.getReader();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        jpg = new Uint8Array([...jpg, ...value]);
        let a = jpg.findIndex((val, i) => val === 0xff && jpg[i + 1] === 0xd8);
        let b = jpg.findIndex((val, i) => val === 0xff && jpg[i + 1] === 0xd9);
        if (a !== -1 && b !== -1) {
          const frameData = jpg.slice(a, b + 2);
          jpg = jpg.slice(b + 2);
          const blob = new Blob([frameData], { type: 'image/jpeg' });
          const frame = await createImageBitmap(blob);
          if (frame) {
            console.log(`Displaying frame ${frameCount} with size: ${frame.width}x${frame.height}...`);
            canvas.width = frame.width;
            canvas.height = frame.height;
            ctx.drawImage(frame, 0, 0);
            frameCount++;
          } else {
            console.log('Failed to decode JPEG frame.');
          }
        }
      }
      console.log('Stream started successfully');
    } catch (error) {
      console.error('Error:', error);
    } finally {
      console.log('Closing the stream...');
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
            <canvas ref={canvasRef} className={styles.video} />
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