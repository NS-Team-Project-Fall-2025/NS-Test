import Link from "next/link";
import { useRouter } from "next/router";
import { useEffect, useState } from "react";

const navItems = [
  { href: "/", label: "Tutor" },
  { href: "/quiz", label: "Quiz Lab" },
  { href: "/knowledge", label: "Knowledge Base" },
  { href: "/sessions", label: "Sessions" }
];

export default function Layout({ children }) {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <div className="app-wrapper">
      <header className="app-header">
        <div className="brand">
          <h1>NetSec Tutor</h1>
          <p className="brand-tagline">
            An easy-to-use tutor that helps you learn network security with clear answers and fun quizzes.
          </p>
        </div>
        <nav>
          <ul>
            {navItems.map((item) => {
              const isActive = mounted && router.pathname === item.href;
              return (
                <li key={item.href} className={isActive ? "active" : ""}>
                  <Link 
                    href={item.href}
                    onClick={(e) => {
                      if (isActive) {
                        e.preventDefault();
                      }
                    }}
                  >
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>
      </header>
      <main className="app-main">{children}</main>
      <footer className="app-footer">
        <span>Fall 2025 · CS 5342 · Project Group 10</span>
      </footer>
    </div>
  );
}
