"""Authentication checks use only the standard library; no OCR/model setup."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "auth_simple", Path(__file__).parents[1] / "src/fraud_scorer/api/auth_simple.py"
)
auth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auth)


class SessionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {
            "FS_AUTH_USERNAME": "test-analyst",
            "FS_AUTH_PASSWORD": "test-only-password",
            "FS_SESSION_SECRET": "test-only-session-secret-at-least-32-characters",
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def issue(self):
        response = Mock()
        auth.mark_authenticated("test-analyst", response)
        self.assertTrue(response.set_cookie.call_args.kwargs["secure"])
        self.assertTrue(response.set_cookie.call_args.kwargs["httponly"])
        return response.set_cookie.call_args.args[1]

    def request(self, token=""):
        return SimpleNamespace(cookies={auth.SESSION_COOKIE: token})

    def test_login_never_authenticates_another_client(self):
        self.assertTrue(auth.authenticate("test-analyst", "test-only-password"))
        logged_in = self.request(self.issue())
        self.assertTrue(auth.is_authenticated(logged_in))
        self.assertFalse(auth.is_authenticated(self.request()))
        self.assertEqual(auth.current_user(logged_in), "test-analyst")
        self.assertIsNone(auth.current_user(self.request()))
        self.assertIsNotNone(auth.last_login(logged_in))
        self.assertIsNone(auth.last_login(self.request()))

    def test_expiry_tampering_and_key_rotation(self):
        with patch.object(auth.time, "time", return_value=1000):
            token = self.issue()
        with patch.object(auth.time, "time", return_value=1001):
            self.assertTrue(auth.is_authenticated(self.request(token)))
            for bad in [token + "x", token + ".extra", "invalid", "☃", "x" * 4097]:
                self.assertFalse(auth.is_authenticated(self.request(bad)))
            with patch.dict("os.environ", {"FS_SESSION_SECRET": "different-test-only-session-secret-at-least-32-characters"}):
                self.assertFalse(auth.is_authenticated(self.request(token)))
        with patch.object(auth.time, "time", return_value=1000 + auth.SESSION_MAX_AGE):
            self.assertFalse(auth.is_authenticated(self.request(token)))
        with patch.object(auth.time, "time", return_value=999):
            self.assertFalse(auth.is_authenticated(self.request(token)))

    def test_missing_configuration_and_wrong_credentials_fail_closed(self):
        token = self.issue()
        self.assertFalse(auth.authenticate("test-analyst", "wrong"))
        for name in ["FS_AUTH_USERNAME", "FS_AUTH_PASSWORD", "FS_SESSION_SECRET"]:
            with patch.dict("os.environ", {name: ""}):
                self.assertFalse(auth.authenticate("test-analyst", "test-only-password"))
                self.assertFalse(auth.is_authenticated(self.request(token)))
                with self.assertRaises(ValueError):
                    self.issue()

    def test_logout_clears_only_the_callers_cookie(self):
        token = self.issue()
        response = Mock()
        auth.clear_authentication(response)
        response.delete_cookie.assert_called_once_with(
            auth.SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="strict"
        )
        self.assertTrue(auth.is_authenticated(self.request(token)))


if __name__ == "__main__":
    unittest.main()
