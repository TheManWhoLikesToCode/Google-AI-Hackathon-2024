import Head from "next/head";
import styles from "../../styles/Home.module.css";

export default function About() {
    return (
        <div className={styles.container}>
            <div className={styles.logoContainer}>
                <img src="/logo.svg" alt="Logo" className={styles.logoImage} />
            </div>
            <div className={styles.header}>
                <a href="/" className={styles.aboutLink}>
                    Home
                </a>
            </div>
            <Head>
                <title>About SafeSteps</title>
                <link rel="icon" href="/favicon.ico" />
            </Head>
            <main>
                <h1 className={styles.title}>About SafeSteps</h1>
                <div className={styles.description}>
                    <p>To Do</p>
                </div>
            </main>
            <footer>
                <a
                    href="https://vercel.com?utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    Powered by{" "}
                    <img src="/vercel.svg" alt="Vercel" className={styles.logo} />
                </a>
            </footer>
        </div>
    );
}
