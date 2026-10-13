/* HOLD8: поток m8 (4 звена + g) с удержанием на пределе |w|=WM — C-версия hold_u/hold_flow из hold8.py (аналог tree_fast, но без разворачивания). */
#include <math.h>
#include <string.h>
#define N 4
static double MS[N], LN[N], GG;
void set_params(const double *ms, const double *ln, double g) { memcpy(MS, ms, sizeof MS); memcpy(LN, ln, sizeof LN); GG = g; }
static void qdd_of(const double *x, const double *u, double *qdd) {
    double th[N], thd[N], M[N][N], r[N], S[N][N], tdd[N]; double s = 0, sd = 0;
    for (int i = 0; i < N; i++) { s += x[i]; sd += x[N + i]; th[i] = s; thd[i] = sd; }
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { int m = i > j ? i : j; double ms = 0; for (int k = m; k < N; k++) ms += MS[k]; S[i][j] = ms * LN[i] * LN[j]; M[i][j] = S[i][j] * cos(th[i] - th[j]); }
    for (int i = 0; i < N; i++) { double ms = 0; for (int k = i; k < N; k++) ms += MS[k]; double v = u[i] - (i + 1 < N ? u[i + 1] : 0.) - GG * LN[i] * ms * cos(th[i]); for (int j = 0; j < N; j++) v -= S[i][j] * sin(th[i] - th[j]) * thd[j] * thd[j]; r[i] = v; }
    for (int c = 0; c < N; c++) { int p = c; for (int i = c + 1; i < N; i++) if (fabs(M[i][c]) > fabs(M[p][c])) p = i;
        if (p != c) { for (int j = 0; j < N; j++) { double t = M[c][j]; M[c][j] = M[p][j]; M[p][j] = t; } double t = r[c]; r[c] = r[p]; r[p] = t; }
        for (int i = c + 1; i < N; i++) { double f = M[i][c] / M[c][c]; for (int j = c; j < N; j++) M[i][j] -= f * M[c][j]; r[i] -= f * r[c]; } }
    for (int i = N - 1; i >= 0; i--) { double v = r[i]; for (int j = i + 1; j < N; j++) v -= M[i][j] * tdd[j]; tdd[i] = v / M[i][i]; }
    qdd[0] = tdd[0]; for (int i = 1; i < N; i++) qdd[i] = tdd[i] - tdd[i - 1];
}
static void f_of(const double *x, const double *u, double *dx) { double q[N]; qdd_of(x, u, q); for (int i = 0; i < N; i++) { dx[i] = x[N + i]; dx[N + i] = q[i]; } }
static void hold_u(const double *x, const double *U, double WM, double H, double *Ue) {
    double a0[N], J[N][N], z[N] = {0}, e[N], q[N]; int Sm[N] = {0};
    qdd_of(x, z, a0);
    for (int j = 0; j < N; j++) { memset(e, 0, sizeof e); e[j] = 1; qdd_of(x, e, q); for (int i = 0; i < N; i++) J[i][j] = q[i] - a0[i]; }
    memcpy(Ue, U, N * sizeof(double));
    for (int it = 0; it < N; it++) {
        double wn[N], tg[N]; int any = 0;
        for (int i = 0; i < N; i++) { double qd = a0[i]; for (int j = 0; j < N; j++) qd += J[i][j] * Ue[j]; wn[i] = x[N + i] + H * qd; tg[i] = (WM * (wn[i] > 0 ? 1. : (wn[i] < 0 ? -1. : 0.)) - x[N + i]) / H; }
        for (int i = 0; i < N; i++) if (!Sm[i] && fabs(wn[i]) > WM) { Sm[i] = 1; any = 1; }
        if (!any) break;
        int idx[N], k = 0; for (int i = 0; i < N; i++) if (Sm[i]) idx[k++] = i;
        double A[N][N + 1];
        for (int a = 0; a < k; a++) { int i = idx[a]; double rhs = tg[i] - a0[i]; for (int j = 0; j < N; j++) if (!Sm[j]) rhs -= J[i][j] * U[j]; for (int b = 0; b < k; b++) A[a][b] = J[i][idx[b]]; A[a][k] = rhs; }
        for (int c = 0; c < k; c++) { int p = c; for (int i = c + 1; i < k; i++) if (fabs(A[i][c]) > fabs(A[p][c])) p = i;
            if (p != c) for (int j = 0; j <= k; j++) { double t = A[c][j]; A[c][j] = A[p][j]; A[p][j] = t; }
            for (int i = c + 1; i < k; i++) { double f = A[i][c] / A[c][c]; for (int j = c; j <= k; j++) A[i][j] -= f * A[c][j]; } }
        double us[N]; for (int i = k - 1; i >= 0; i--) { double v = A[i][k]; for (int j = i + 1; j < k; j++) v -= A[i][j] * us[j]; us[i] = v / A[i][i]; }
        for (int a = 0; a < k; a++) { double v = us[a]; Ue[idx[a]] = v > 1 ? 1 : (v < -1 ? -1 : v); }
    }
}
static void rk4(const double *x, const double *u, double h, double *o) {
    double k1[2 * N], k2[2 * N], k3[2 * N], k4[2 * N], y[2 * N];
    f_of(x, u, k1); for (int i = 0; i < 2 * N; i++) y[i] = x[i] + h / 2 * k1[i];
    f_of(y, u, k2); for (int i = 0; i < 2 * N; i++) y[i] = x[i] + h / 2 * k2[i];
    f_of(y, u, k3); for (int i = 0; i < 2 * N; i++) y[i] = x[i] + h * k3[i];
    f_of(y, u, k4); for (int i = 0; i < 2 * N; i++) o[i] = x[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
}
/* B состояний X[B*8], слоёв U[B*4], время t, k подшагов (k=round(t/H)); выход: Xout, Xhalf (после k/2-1 подшага+1), rec[B*k*4] (эффективные u) либо NULL */
void hold_flow(int B, const double *X, const double *U, double t, int k, double WM, double H, double *Xout, double *Xhalf, double *rec) {
    double h = t / k;
    for (int b = 0; b < B; b++) {
        double x[2 * N], y[2 * N], Ue[N]; memcpy(x, X + b * 2 * N, sizeof x); memcpy(Xhalf + b * 2 * N, x, sizeof x);
        for (int s = 0; s < k; s++) {
            hold_u(x, U + b * N, WM, H, Ue); if (rec) memcpy(rec + ((long)b * k + s) * N, Ue, sizeof Ue);
            rk4(x, Ue, h, y); memcpy(x, y, sizeof x); if (s == k / 2 - 1) memcpy(Xhalf + b * 2 * N, x, sizeof x);
        }
        memcpy(Xout + b * 2 * N, x, sizeof x);
    }
}
