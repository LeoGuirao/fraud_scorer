# Web authentication

Set FS_AUTH_USERNAME, FS_AUTH_PASSWORD and FS_SESSION_SECRET through protected
deployment configuration. Use a new strong password and a new random session
secret of at least 32 characters. Do not reuse the credentials previously
committed to the repository, and do not commit real configuration values.
Missing configuration denies authentication.

Serve the web UI over HTTPS. Session cookies are Secure, HttpOnly and SameSite
Strict, and expire after eight hours. The same session secret must be supplied
to every worker. Rotating it invalidates all existing sessions. Logging out
removes that browser's cookie; a copied cookie remains valid until expiry or
secret rotation. This replaces the old process-wide login flag.

For a reverse proxy, terminate HTTPS at the proxy and forward to the web server
on a private/local interface. Cookie security depends on the browser-facing
HTTPS connection; direct HTTP access on a LAN will not establish a session.

The documented launcher runs fraud_scorer.api.web_interface:app. The separate,
unreferenced legacy fraud_scorer.api.main:app does not use this web authentication
and must not be exposed to untrusted clients. Keep it unserved or behind an
independent authenticated gateway.
