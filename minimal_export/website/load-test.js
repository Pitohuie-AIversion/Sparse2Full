import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 200 }, // ramp up to 200 users
    { duration: '1m', target: 1000 }, // ramp up to 1000 users
    { duration: '30s', target: 0 },   // ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(99)<500'], // 99% of requests must complete below 500ms
  },
};

export default function () {
  // Assuming the website will be deployed to this URL
  const res = http.get('https://example.github.io/Sparse2Full/');
  check(res, { 'status was 200': (r) => r.status == 200 });
  sleep(1);
}