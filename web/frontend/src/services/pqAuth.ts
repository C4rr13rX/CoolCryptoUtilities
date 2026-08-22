/**
 * Quantum-safe password login — browser side.
 *
 * Must stay byte-for-byte compatible with `serverless/hybrid/pq_auth.py`.
 * The four things both sides have to agree on exactly:
 *
 *   1. ML-KEM-768 (FIPS 203) for encapsulation.
 *   2. HKDF-SHA-384, salt `coolcrypto-pq-auth-v1`, info = transcript, 32 bytes.
 *   3. AES-256-GCM with a 12-byte nonce and the transcript as associated data.
 *   4. transcript = utf8(`${challenge_id}|${server_key_b64}`).
 *
 * Change any of those and login fails closed with "could not open credential
 * envelope" — the server refuses to guess.
 *
 * Why the password is encrypted when TLS already encrypts it: TLS 1.3 key
 * exchange is X25519/ECDHE, which a quantum computer breaks. An adversary
 * recording traffic today can decrypt it later and recover a password that is
 * probably still valid. ML-KEM inside the tunnel means the recorded bytes stay
 * useless. See the module docstring in pq_auth.py for the full threat model.
 */

import { ml_kem768 } from '@noble/post-quantum/ml-kem.js';
import { hkdf } from '@noble/hashes/hkdf.js';
import { sha384 } from '@noble/hashes/sha2.js';
import { gcm } from '@noble/ciphers/aes.js';

export interface Challenge {
  challenge_id: string;
  server_key: string;
  kem: string;
  expires_in: number;
}

export interface SessionUser {
  id: number;
  email: string;
}

export interface LoginResult {
  token: string;
  expires_at: number;
  user: SessionUser;
}

const TOKEN_KEY = 'ccu.auth.token';

/* ---------------------------------------------------------------- base64 */
function b64encode(raw: Uint8Array): string {
  let s = '';
  for (const byte of raw) s += String.fromCharCode(byte);
  return btoa(s);
}

function b64decode(text: string): Uint8Array {
  const bin = atob(text);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
  return out;
}

/* ------------------------------------------------------------- transcript */
/** Binds the derived key to this attempt, so an envelope cannot be replayed. */
function transcriptFor(challengeId: string, serverKeyB64: string): Uint8Array {
  return new TextEncoder().encode(`${challengeId}|${serverKeyB64}`);
}

/* ------------------------------------------------------------------ seal */
export function sealPassword(
  password: string,
  serverKeyB64: string,
  transcript: Uint8Array,
): { kem_ct: string; nonce: string; sealed: string } {
  const { cipherText, sharedSecret } = ml_kem768.encapsulate(b64decode(serverKeyB64));

  // Must match Python's HKDF(SHA384, length=32, salt=..., info=transcript).
  const key = hkdf(
    sha384,
    sharedSecret,
    new TextEncoder().encode('coolcrypto-pq-auth-v1'),
    transcript,
    32,
  );

  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const sealed = gcm(key, nonce, transcript).encrypt(
    new TextEncoder().encode(password),
  );

  return {
    kem_ct: b64encode(cipherText),
    nonce: b64encode(nonce),
    sealed: b64encode(sealed),
  };
}

/* ------------------------------------------------------------------ flows */
export class PqAuthClient {
  constructor(private readonly base: string = '/api/auth') {}

  private async post<T>(path: string, body?: unknown, token?: string): Promise<T> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const res = await fetch(`${this.base}${path}`, {
      method: 'POST',
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: 'include',
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error((data as any)?.error || `request failed (${res.status})`);
    }
    return data as T;
  }

  /**
   * Two-step login: fetch a fresh challenge, seal the password to it, submit.
   *
   * The password is encrypted in this function and the plaintext never leaves
   * it — do not log, store, or pass `password` anywhere else.
   */
  async login(email: string, password: string): Promise<LoginResult> {
    const challenge = await this.post<Challenge>('/challenge');
    const transcript = transcriptFor(challenge.challenge_id, challenge.server_key);
    const envelope = sealPassword(password, challenge.server_key, transcript);

    const result = await this.post<LoginResult>('/login', {
      challenge_id: challenge.challenge_id,
      email,
      envelope,
    });
    localStorage.setItem(TOKEN_KEY, result.token);
    return result;
  }

  async logout(): Promise<void> {
    const token = this.token();
    if (token) {
      // Best-effort: the local token is cleared regardless, so a network
      // failure cannot strand the user in a logged-in-looking state.
      await this.post('/logout', undefined, token).catch(() => undefined);
    }
    localStorage.removeItem(TOKEN_KEY);
  }

  async session(): Promise<SessionUser | null> {
    const token = this.token();
    if (!token) return null;
    try {
      const res = await fetch(`${this.base}/session`, {
        headers: { Authorization: `Bearer ${token}` },
        credentials: 'include',
      });
      if (!res.ok) {
        // Expired or revoked: drop it so the UI shows the login form.
        localStorage.removeItem(TOKEN_KEY);
        return null;
      }
      const data = await res.json();
      return data.user as SessionUser;
    } catch {
      return null;
    }
  }

  token(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }
}

export const pqAuth = new PqAuthClient();
