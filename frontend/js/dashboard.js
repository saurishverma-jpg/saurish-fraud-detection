// Dashboard charts and live simulation

const chartDefaults = {
  color: 'rgba(255,255,255,0.7)',
  borderColor: 'rgba(255,255,255,0.1)',
};

// Bar Chart
const barCtx = document.getElementById('barChart');
if (barCtx) {
  new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
      datasets: [
        {
          label: 'Legitimate',
          data: [1820, 1950, 1700, 2100, 1890, 1400, 1620],
          backgroundColor: 'rgba(29,158,117,0.5)',
          borderRadius: 6,
        },
        {
          label: 'Fraudulent',
          data: [12, 8, 15, 6, 18, 5, 9],
          backgroundColor: 'rgba(226,75,74,0.6)',
          borderRadius: 6,
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: 'rgba(255,255,255,0.6)', font: { size: 12 } } }
      },
      scales: {
        x: { ticks: { color: 'rgba(255,255,255,0.4)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: 'rgba(255,255,255,0.4)' }, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });
}

// Doughnut Chart
const doughCtx = document.getElementById('doughnutChart');
if (doughCtx) {
  new Chart(doughCtx, {
    type: 'doughnut',
    data: {
      labels: ['Card Fraud','Phishing','Wire Fraud','Account Takeover','Other'],
      datasets: [{
        data: [35, 25, 20, 15, 5],
        backgroundColor: [
          'rgba(226,75,74,0.7)',
          'rgba(239,159,39,0.7)',
          'rgba(127,119,221,0.7)',
          'rgba(29,158,117,0.7)',
          'rgba(255,255,255,0.2)',
        ],
        borderWidth: 0,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: 'rgba(255,255,255,0.6)', font: { size: 11 }, padding: 12 }
        }
      }
    }
  });
}

// Simulate live count updates
function simulateLive() {
  const totalEl = document.getElementById('totalTx');
  const fraudEl = document.getElementById('fraudCount');
  if (!totalEl || !fraudEl) return;

  setInterval(() => {
    let total = parseInt(totalEl.innerText.replace(/,/g,''));
    let fraud = parseInt(fraudEl.innerText);
    total += Math.floor(Math.random() * 3);
    if (Math.random() < 0.15) fraud += 1;
    totalEl.innerText = total.toLocaleString();
    fraudEl.innerText = fraud;
  }, 3000);
}

simulateLive();
