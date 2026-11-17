import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CampusAsk | University of Central Missouri",
  description:
    "Official UCM policy answers, grounded in the undergraduate and graduate catalogs.",
};

/** Keep CampusAsk on the light UCM brand palette even when the OS prefers dark. */
export const viewport: Viewport = {
  colorScheme: "light",
  themeColor: "#ffffff",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // Browser extensions (e.g. Scribe's data-scribe-recorder-ready) add attributes to
    // <html> before React hydrates. This only ignores attribute mismatches on this one
    // element; mismatches anywhere inside the page are still reported.
    <html lang="en" data-theme="light" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
