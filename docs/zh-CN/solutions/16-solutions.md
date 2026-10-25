# 第 16 章参考答案

[返回第 16 章](../16-vae-and-diffusion.md) | [English](../../en/solutions/16-solutions.md)

## 1. ELBO

插入任意 $q(z\mid x)$：
$\log p(x)=\log\mathbb E_q[p(x,z)/q(z\mid x)]\ge
\mathbb E_q\log p(x\mid z)-D_{KL}(q\|p)$。差值恰为
$D_{KL}(q(z\mid x)\|p(z\mid x))\ge0$。

## 2. 对角高斯 KL

把高斯密度对数比取 $q$ 下期望，使用
$\mathbb E_q z_j=\mu_j$、 $\mathbb E_q(z_j-\mu_j)^2=\sigma_j^2$，得到
$\frac12\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1)$。

## 3–4. 重参数化与 VAE 诊断

用 $z=\mu+\exp(0.5\log\sigma^2)\epsilon$。固定 $\epsilon$ 做中心差分，与自动梯度比较相对误差。
训练时分别画重构与 KL；固定输入改变 $z$、固定 $z$ 改变单维，并从先验采样。KL 近零且输出几乎
不随 $z$ 变是 posterior collapse 证据。

## 5. 扩散闭式

递推代入 $x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t$。独立高斯线性组合仍为高斯，
信号系数相乘为 $\sqrt{\bar\alpha_t}$，总噪声方差为 $1-\bar\alpha_t$。

## 6–7. 采样与审计

固定模型与随机种子，联合扫日程、采样器、步数和 guidance，报告质量、覆盖、条件一致性、P95 延迟。
另做训练样本最近邻、成员推断/重复、类别与群体覆盖、提示敏感性审计；单一 FID 不足以验收。
