import Link from "next/link";
import AskLauncher from "@/components/AskLauncher";
import SiteHeader from "@/components/SiteHeader";
import { SAMPLE_GROUPS } from "@/lib/api";
import styles from "./page.module.css";

export default function LandingPage() {
  return (
    <div className={styles.page}>
      <SiteHeader active="home" />

      <main>
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <div className={styles.redBox}>
              <p className={styles.eyebrow}>Academic Catalog Desk</p>
              <h1 className={styles.title}>CampusAsk</h1>
              <p className={styles.lede}>
                Answers to UCM academic policy questions, taken from the 2026–27 Undergraduate and
                Graduate Catalogs, with the page each answer came from.
              </p>
              <Link className={styles.askBtn} href="/ask">
                <svg className={styles.askIcon} viewBox="0 0 24 24" aria-hidden="true">
                  <circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" strokeWidth="2.4" />
                  <path d="M15.5 15.5 21 21" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
                </svg>
                Ask a Question
              </Link>
            </div>
          </div>
        </section>

        <section className={styles.topics}>
          <div className={styles.inner}>
            <h2 className={styles.sectionTitle}>What you can ask</h2>
            <div className={styles.topicGrid}>
              {SAMPLE_GROUPS.map((group) => (
                <div key={group.topic} className={styles.topic}>
                  <h3>{group.topic}</h3>
                  <ul>
                    {group.questions.map((q) => (
                      <li key={q.id}>{q.question}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
            <ol className={styles.steps}>
              <li>
                <strong>Search</strong> CampusAsk finds the catalog pages that match your question.
              </li>
              <li>
                <strong>Answer</strong> It writes an answer using only those pages.
              </li>
              <li>
                <strong>Cite</strong> Every answer links to the catalog page it came from. If the
                catalogs don&apos;t cover your question, it tells you.
              </li>
            </ol>
          </div>
        </section>
      </main>

      <footer className={styles.footer}>
        <div className={styles.inner}>
          <div>
            <p className={styles.footerHead}>University of Central Missouri</p>
            <p>Warrensburg, Missouri</p>
          </div>
          <div>
            <p className={styles.footerHead}>Official sources</p>
            <p>
              <a href="https://catalog.ucmo.edu/" target="_blank" rel="noopener noreferrer">
                UCM Academic Catalogs
              </a>
            </p>
            <p>
              <a
                href="https://www.ucmo.edu/current-students/office-of-the-registrar-and-student-records/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Office of the Registrar
              </a>
            </p>
          </div>
        </div>
      </footer>

      <AskLauncher />
    </div>
  );
}
