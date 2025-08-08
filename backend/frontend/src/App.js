import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [userRole, setUserRole] = useState("");
  const [industry, setIndustry] = useState("");
  const [topic, setTopic] = useState("");
  const [post, setPost] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [resumeAnalysis, setResumeAnalysis] = useState("");

  const [trends, setTrends] = useState([]);
  const [trendLoading, setTrendLoading] = useState(false);

  const [strategy, setStrategy] = useState("");
  const [strategyLoading, setStrategyLoading] = useState(false);

  const [analytics, setAnalytics] = useState(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);

  const [postType, setPostType] = useState("article");
  const [advancedContent, setAdvancedContent] = useState("");
  const [advancedLoading, setAdvancedLoading] = useState(false);

  const [resumeHistory, setResumeHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [token, setToken] = useState(localStorage.getItem("token") || "");
  const [authMode, setAuthMode] = useState("login");

  useEffect(() => {
    document.body.className = darkMode ? "dark" : "light";
  }, [darkMode]);

  const authHeaders = { Authorization: `Bearer ${token}` };

  // Auth
  const handleAuth = async () => {
    if (!username || !password) return alert("Please fill in all fields");
    try {
      if (authMode === "signup") {
        await axios.post("http://127.0.0.1:8000/signup/", { username, password });
        alert("✅ Signup successful! Please log in.");
        setAuthMode("login");
      } else {
        const res = await axios.post("http://127.0.0.1:8000/login/", { username, password });
        localStorage.setItem("token", res.data.access_token);
        setToken(res.data.access_token);
        alert("✅ Logged in!");
        fetchResumeHistory();
      }
    } catch {
      alert("❌ Authentication failed");
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    setToken("");
    setUsername("");
    setPassword("");
    setResumeHistory([]);
    setPost("");
    setResumeAnalysis("");
  };

  // Resume upload
  const handleResumeUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    setUploading(true);
    try {
      const response = await axios.post("http://127.0.0.1:8000/upload_resume/", formData, {
        headers: { ...authHeaders, "Content-Type": "multipart/form-data" },
      });
      const analysis = response.data.analysis || "⚠️ No insights extracted.";
      setResumeAnalysis(analysis);
      fetchResumeHistory();
      const lower = analysis.toLowerCase();
      if (lower.includes("data scientist")) setUserRole("Data Scientist");
      else if (lower.includes("software engineer")) setUserRole("Software Engineer");
      else if (lower.includes("marketing")) setUserRole("Marketing Executive");
      else if (lower.includes("designer")) setUserRole("UI/UX Designer");
    } catch {
      setResumeAnalysis("❌ Failed to analyze resume.");
    }
    setUploading(false);
  };

  // Resume history
  const fetchResumeHistory = async () => {
    if (!token) return;
    setHistoryLoading(true);
    try {
      const response = await axios.get("http://127.0.0.1:8000/resumes/", { headers: authHeaders });
      setResumeHistory(response.data || []);
    } catch {
      setResumeHistory([]);
    }
    setHistoryLoading(false);
  };

  useEffect(() => {
    if (token) fetchResumeHistory();
  }, [token]);

  // LinkedIn post
  const generatePost = async () => {
    setLoading(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/generate_post/",
        { user_role: userRole, industry, topic },
        { headers: authHeaders }
      );
      setPost(response.data.post || response.data.error);
    } catch {
      setPost("❌ Error generating post. Is your backend running?");
    }
    setLoading(false);
  };

  // Advanced post
  const generateAdvancedContent = async () => {
    if (!userRole || !industry || !topic || !postType) return alert("All fields are required!");
    setAdvancedLoading(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/generate_advanced_content/",
        { user_role: userRole, industry, topic, post_type: postType },
        { headers: authHeaders }
      );
      setAdvancedContent(response.data.content || "❌ Failed to generate advanced content.");
    } catch {
      setAdvancedContent("❌ Error connecting to server.");
    }
    setAdvancedLoading(false);
  };

  // Trends
  const fetchTrends = async () => {
    if (!industry) return alert("Please enter an industry!");
    setTrendLoading(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/industry_trends/",
        { industry },
        { headers: authHeaders }
      );
      setTrends(response.data.trends || []);
    } catch {
      setTrends(["❌ Failed to fetch trends"]);
    }
    setTrendLoading(false);
  };

  // Strategy
  const fetchStrategy = async () => {
    if (!userRole || !industry) return alert("Please enter role and industry!");
    setStrategyLoading(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/content_strategy/",
        { user_role: userRole, industry },
        { headers: authHeaders }
      );
      setStrategy(response.data.strategy || "⚠️ No suggestions available.");
    } catch {
      setStrategy("❌ Failed to fetch strategy.");
    }
    setStrategyLoading(false);
  };

  // Analytics (fixed)
  const fetchAnalytics = async () => {
    setAnalyticsLoading(true);
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/mock_analytics/",
        {},
        { headers: authHeaders } // ✅ Added Bearer token
      );
      setAnalytics(response.data.analytics);
    } catch {
      setAnalytics(null);
      alert("❌ Failed to fetch analytics");
    }
    setAnalyticsLoading(false);
  };

  // Auth screen
  if (!token) {
    return (
      <div className="auth-container">
        <h1>Influence OS - {authMode === "login" ? "Login" : "Signup"}</h1>
        <input type="text" placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} />
        <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
        <button onClick={handleAuth}>{authMode === "login" ? "Login" : "Signup"}</button>
        <p onClick={() => setAuthMode(authMode === "login" ? "signup" : "login")} style={{ cursor: "pointer" }}>
          {authMode === "login" ? "Need an account? Sign up" : "Have an account? Log in"}
        </p>
      </div>
    );
  }

  // Main app UI
  return (
    <div className={`App ${darkMode ? "dark" : "light"}`}>
      <div className="header">
        <h1 className="brand-title">Influence OS</h1>
        <label className="toggle">
          🌙 Dark Mode
          <input type="checkbox" checked={darkMode} onChange={() => setDarkMode(!darkMode)} />
        </label>
        <button onClick={handleLogout}>🚪 Logout</button>
      </div>

      {/* Resume Upload */}
      <section className="section-box">
        <h2>📄 Upload Resume (PDF)</h2>
        <input type="file" accept=".pdf" onChange={handleResumeUpload} />
        {uploading && <p>⏳ Uploading and analyzing resume...</p>}
        {resumeAnalysis && (
          <div className="output">
            <h4>🧠 Resume Analysis:</h4>
            <textarea readOnly value={resumeAnalysis} />
          </div>
        )}
      </section>

      {/* Resume History */}
      <section className="section-box">
        <h2>📜 Resume History</h2>
        <button onClick={fetchResumeHistory} disabled={historyLoading}>
          {historyLoading ? "⏳ Loading..." : "Refresh History"}
        </button>
        {resumeHistory.length > 0 ? (
          <table className="history-table">
            <thead>
              <tr>
                <th>File Name</th>
                <th>Uploaded</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {resumeHistory.map((res) => (
                <tr key={res.id}>
                  <td>{res.filename}</td>
                  <td>{new Date(res.uploaded_at).toLocaleString()}</td>
                  <td>
                    <button onClick={() => setResumeAnalysis(res.analysis)}>View</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>⚠️ No resumes found.</p>
        )}
      </section>

      {/* LinkedIn Post Generator */}
      <section className="section-box">
        <h2>📝 LinkedIn Post Generator</h2>
        <input type="text" placeholder="Your Role" value={userRole} onChange={(e) => setUserRole(e.target.value)} />
        <input type="text" placeholder="Industry" value={industry} onChange={(e) => setIndustry(e.target.value)} />
        <input type="text" placeholder="Topic" value={topic} onChange={(e) => setTopic(e.target.value)} />
        <button onClick={generatePost} disabled={loading}>
          {loading ? "✍️ Generating..." : "Generate Post"}
        </button>
        {post && (
          <div className="output">
            <h3>✅ Generated Post:</h3>
            <textarea readOnly value={post} />
          </div>
        )}
      </section>

      {/* Advanced Post */}
      <section className="section-box">
        <h2>🧠 Advanced Post by Type</h2>
        <select value={postType} onChange={(e) => setPostType(e.target.value)}>
          <option value="article">Article</option>
          <option value="update">Update</option>
          <option value="carousel">Carousel</option>
        </select>
        <button onClick={generateAdvancedContent} disabled={advancedLoading}>
          {advancedLoading ? "⚙️ Generating..." : "Generate Advanced Content"}
        </button>
        {advancedContent && (
          <div className="card-output">
            <pre>{advancedContent}</pre>
          </div>
        )}
      </section>

      {/* Industry Trends */}
      <section className="section-box">
        <h2>🌐 Industry Trends</h2>
        <button onClick={fetchTrends} disabled={trendLoading}>
          {trendLoading ? "🔎 Fetching..." : "Show Latest Trends"}
        </button>
        {trends.length > 0 && (
          <div className="trends-grid">
            {trends.map((trend, index) => (
              <div className="trend-card" key={index}>
                🔸 {trend}
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Content Strategy */}
      <section className="section-box">
        <h2>📢 Content Strategy Recommendations</h2>
        <button onClick={fetchStrategy} disabled={strategyLoading}>
          {strategyLoading ? "📊 Analyzing..." : "Show Strategy"}
        </button>
        {strategy && (
          <div className="strategy-grid">
            {strategy.split("\n").map((line, index) => (
              <div className="strategy-card" key={index}>
                {line}
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Analytics */}
      <section className="section-box">
        <h2>📈 Post Performance Analytics</h2>
        <button onClick={fetchAnalytics} disabled={analyticsLoading}>
          {analyticsLoading ? "📊 Loading..." : "Show Analytics"}
        </button>
        {analytics && (
          <div className="analytics-grid">
            <div className="analytics-card">
              <span className="icon">👁️</span>
              <div className="metric">{analytics.views}</div>
              <div className="label">Views</div>
            </div>
            <div className="analytics-card">
              <span className="icon">❤️</span>
              <div className="metric">{analytics.likes}</div>
              <div className="label">Likes</div>
            </div>
            <div className="analytics-card">
              <span className="icon">💬</span>
              <div className="metric">{analytics.comments}</div>
              <div className="label">Comments</div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

export default App;
