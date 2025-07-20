---
vssue: ""
---

# 博客文章

<div class="intro-container">
  <div class="intro-text">
    <span class="greeting">您好，很高兴认识你 <span class="wave">👋</span></span>
    <span class="name">我是 <span class="highlight">KenForever1</span></span>
    <span class="name">能同途偶遇在这星球上，探索、记录、享受…</span>
  </div>
</div>

<style>
.intro-container {
  background: linear-gradient(145deg, rgba(255,255,255,0.8) 0%, rgba(240,240,240,0.6) 100%);
  border-radius: 16px;
  padding: 2rem;
  margin: 2rem 0;
  box-shadow: 0 4px 20px rgba(0,0,0,0.05);
  border: 1px solid rgba(200,200,200,0.2);
  transition: all 0.3s ease;
}

.intro-container:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.intro-text {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.greeting, .name {
  display: block;
  font-size: 1.0rem;
  line-height: 1.6;
  color: #555;
  /* font-family: 'LXGW WenKai', sans-serif; */
}

.highlight {
  background: linear-gradient(120deg,rgb(96, 189, 127) 0%,rgb(135, 215, 123) 100%);
  background-clip: text;
  -webkit-background-clip: text;
  color: transparent;
  font-weight: bold;
  padding: 0 0.2rem;
  position: relative;
}

.wave {
  display: inline-block;
  animation: wave 1.5s infinite;
  transform-origin: 70% 70%;
}

@keyframes wave {
  0% { transform: rotate(0deg); }
  10% { transform: rotate(14deg); }
  20% { transform: rotate(-8deg); }
  30% { transform: rotate(14deg); }
  40% { transform: rotate(-4deg); }
  50% { transform: rotate(10deg); }
  60% { transform: rotate(0deg); }
  100% { transform: rotate(0deg); }
}

/* 深色模式适配 */
[data-md-color-scheme="slate"] .intro-container {
  background: linear-gradient(145deg, rgba(31,33,40) 0%, rgba(31,33,40) 100%);
  border: 1px solid rgba(80,80,80,0.2);
}

[data-md-color-scheme="slate"] .greeting, 
[data-md-color-scheme="slate"] .name {
  color: #e0e0e0;
}

[data-md-color-scheme="slate"] .highlight {
  background: linear-gradient(120deg, #7BA7D7 0%, #A8C5E5 100%);
  background-clip: text;
  -webkit-background-clip: text;
}

/* 移动端适配 */
@media (max-width: 768px) {
  .intro-container {
    padding: 1.5rem;
    margin: 1.5rem 0;
  }
  
  .greeting, .name {
    font-size: 1.3rem;
  }
}
</style>
