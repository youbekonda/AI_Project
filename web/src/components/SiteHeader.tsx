import Image from "next/image";
import Link from "next/link";
import { ReactNode } from "react";
import styles from "./SiteHeader.module.css";

/** Red utility links, as in the www.ucmo.edu top bar — pointed at what catalog users need. */
const UTILITY_LINKS = [
  { label: "Catalog", href: "https://catalog.ucmo.edu/" },
  {
    label: "Registrar",
    href: "https://www.ucmo.edu/current-students/office-of-the-registrar-and-student-records/",
  },
  { label: "MyCentral", href: "https://mycentral.ucmo.edu/" },
  { label: "UCMO.edu", href: "https://www.ucmo.edu/" },
];

type Active = "home" | "ask";

export default function SiteHeader({
  active,
  navStart,
  fluid = false,
}: {
  active: Active;
  /** Rendered at the start of the charcoal nav bar (e.g. the Ask page's menu button on phones). */
  navStart?: ReactNode;
  /** Full-width header, so it lines up with an app layout that has a left panel. */
  fluid?: boolean;
}) {
  const navItems: { key: Active; label: string; href: string }[] = [
    { key: "home", label: "Home", href: "/" },
    { key: "ask", label: "Ask a Question", href: "/ask" },
  ];

  return (
    <header className={`${styles.header} ${fluid ? styles.fluid : ""}`}>
      <div className={styles.top}>
        <div className={styles.inner}>
          <Link href="/" className={styles.logoLink} aria-label="CampusAsk home">
            <Image
              className={styles.logo}
              src="/ucm/ucm-logo-header.png"
              alt="University of Central Missouri"
              width={460}
              height={170}
              priority
            />
          </Link>
          <ul className={styles.utility} aria-label="UCM resources">
            {UTILITY_LINKS.map((link) => (
              <li key={link.label}>
                <a href={link.href} target="_blank" rel="noopener noreferrer">
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <nav className={styles.nav} aria-label="CampusAsk">
        <div className={styles.inner}>
          {navStart}
          <span className={styles.product}>CampusAsk</span>
          <ul className={styles.navList}>
            {navItems.map((item) => (
              <li key={item.key}>
                <Link
                  href={item.href}
                  className={active === item.key ? styles.navActive : undefined}
                  aria-current={active === item.key ? "page" : undefined}
                >
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </nav>
    </header>
  );
}
