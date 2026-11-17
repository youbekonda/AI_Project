import Link from "next/link";
import styles from "./AskLauncher.module.css";

/** Floating "Ask a question" launcher with Mo the Mule, pinned bottom-right. */
export default function AskLauncher({ href = "/ask" }: { href?: string }) {
  return (
    <Link href={href} className={styles.launcher} aria-label="Ask a question with CampusAsk">
      <span className={styles.label}>Ask a question</span>
      <span className={styles.avatar} aria-hidden="true" />
    </Link>
  );
}
