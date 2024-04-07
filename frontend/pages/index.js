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

  const startStream = () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          setIsStreaming(true);
        })
        .catch((error) => {
          console.error('Error accessing camera:', error);
        });
    }
  };

  const stopStream = () => {
    const stream = videoRef.current.srcObject;
    if (stream) {
      const tracks = stream.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
  };

  const handleUpload = (event) => {
    const file = event.target.files[0];
    // Handle the uploaded file here
    console.log('Uploaded file:', file);
  };

  const handleRedirect = () => {
    window.location.href = "/";
  };

  return (
    <div className={styles.container}>

      <div className={styles.logoContainer}>
        <img src="/logo.svg" alt="Logo" className={styles.logoImage} />
      </div>

      <Head>
        <title>Demo Page</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <sidenav>
        <a href="/">Home</a>
        <a href="/about">About</a>
        <a href="/contact">Contact</a>
        <button onClick={handleOpenDialog} className={styles.button}>
          Account
        </button>

        {isDialogOpen && (
          <div className={styles.dialog}>
            <h3>Create Account</h3>
            <form>
              <label htmlFor="email">Email:</label>
              <input type="email" id="email" />

              <label htmlFor="password">Password:</label>
              <input type="password" id="password" />

              <button type="submit">Create Account</button>
            </form>

            <h3>Sign In</h3>
            <form>
              <label htmlFor="email">Email:</label>
              <input type="email" id="email" />

              <label htmlFor="password">Password:</label>
              <input type="password" id="password" />

              <button type="submit">Sign In</button>
            </form>
          </div>
        )}
        
      </sidenav>

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

      <style jsx>{`
        main {
          padding: 5rem 0;
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }
        footer {
          width: 100%;
          height: 100px;
          border-top: 1px solid #eaeaea;
          display: flex;
          justify-content: center;
          align-items: center;
        }
        footer img {
          margin-left: 0.5rem;
        }
        footer a {
          display: flex;
          justify-content: center;
          align-items: center;
          text-decoration: none;
          color: inherit;
        }
        .video {
          width: 100%;
          max-width: 640px;
          height: auto;
        }
      `}</style>

      <style jsx global>{`
        html,
        body {
          padding: 0;
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto,
            Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue,
            sans-serif;
        }
        * {
          box-sizing: border-box;
        }
      `}</style>
    </div>
  );
}