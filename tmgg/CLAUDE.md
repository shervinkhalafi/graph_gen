Unless explicitly specified by the user or confirmed with a question, we never silence pyright errors or warnings, and we never do "graceful fallback", we always _address_ the error/warning, and we always fail loudly and informatively with an exception

