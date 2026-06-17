import { useState } from "react";
import type { FormEvent } from "react";
import { api, ApiError } from "../api";

export function Login({ onSuccess }: { onSuccess: () => void }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.login(password);
      onSuccess();
    } catch (err) {
      if (err instanceof ApiError && err.status === 429)
        setError("Too many attempts — wait a few minutes.");
      else setError("Invalid password.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login">
      <form className="card login-card" onSubmit={submit}>
        <h1>🎯 Sports Edge</h1>
        <p className="muted">Enter the access password.</p>
        <input
          type="password"
          value={password}
          autoFocus
          placeholder="Password"
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <div className="error">{error}</div>}
        <button type="submit" disabled={busy || !password}>
          {busy ? "…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
