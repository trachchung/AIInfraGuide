import type { CollectionEntry } from 'astro:content';
import type { Lang } from '../i18n/ui';

type Interview = CollectionEntry<'interview'>;

export interface CompanyGroup {
  company: string;
  articles: Interview[];
}

export interface TierGroup {
  tier: string;
  label: string;
  color: string;
  companies: CompanyGroup[];
}

export const TIER_META: Record<string, { label: string; color: string }> = {
  T0: { label: 'T0 大厂', color: '#EF4444' },
  T1: { label: 'T1 大厂/独角兽', color: '#F59E0B' },
  T2: { label: 'T2 AI 独角兽', color: '#8B5CF6' },
  T3: { label: 'T3 芯片/硬件', color: '#06B6D4' },
  T4: { label: 'T4 车企/自驾', color: '#10B981' },
  T5: { label: 'T5 其他', color: '#6B7280' },
  综合: { label: '综合面经', color: '#3B82F6' },
};

const TIER_ORDER = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', '综合'];

/**
 * Group interview entries by tier → company.
 * Returns an array of TierGroup sorted by tier order,
 * each containing CompanyGroups sorted by article count descending.
 */
export function groupByCompany(items: Interview[]): TierGroup[] {
  const sorted = [...items].sort((a, b) => a.data.order - b.data.order);

  // tier → company → articles
  const tierMap = new Map<string, Map<string, Interview[]>>();

  for (const item of sorted) {
    const tier = item.data.tier;
    const company = item.data.company;
    if (!tierMap.has(tier)) tierMap.set(tier, new Map());
    const companyMap = tierMap.get(tier)!;
    if (!companyMap.has(company)) companyMap.set(company, []);
    companyMap.get(company)!.push(item);
  }

  const result: TierGroup[] = [];

  for (const tier of TIER_ORDER) {
    const companyMap = tierMap.get(tier);
    if (!companyMap) continue;

    const meta = TIER_META[tier] ?? { label: tier, color: '#6B7280' };
    const companies: CompanyGroup[] = [...companyMap.entries()]
      .map(([company, articles]) => ({ company, articles }))
      .sort((a, b) => b.articles.length - a.articles.length);

    result.push({ tier, ...meta, companies });
  }

  return result;
}

/**
 * Flatten all interviews into reading order: sorted by tier order, then by `order` field.
 */
export function flattenCompanyOrder(items: Interview[]): Interview[] {
  return [...items].sort((a, b) => a.data.order - b.data.order);
}

/**
 * Latin-script names for companies, used on the English and Vietnamese pages.
 * Pinyin is used where no official English name is known. Unlisted names are shown as-is.
 */
const COMPANY_NAMES: Record<string, string | Partial<Record<Lang, string>>> = {
  百度: 'Baidu', 阿里巴巴: 'Alibaba', 字节跳动: 'ByteDance', 快手: 'Kuaishou', 腾讯: 'Tencent',
  蔚来: 'NIO', 小米: 'Xiaomi', 卓驭: 'Zhuoyu', 蚂蚁: 'Ant Group', 美团: 'Meituan',
  科大讯飞: 'iFlytek', 理想汽车: 'Li Auto', 海康威视: 'Hikvision', 小鹏汽车: 'XPeng', 太初: 'Taichu',
  三星: 'Samsung', 飞腾: 'Phytium', 荣耀: 'Honor', 英伟达: 'NVIDIA', 沐曦: 'MetaX',
  智源研究院: 'BAAI', 旷视科技: 'Megvii', 文远知行: 'WeRide', 壁仞科技: 'Biren Technology', 华为: 'Huawei',
  元戎启行: 'DeepRoute.ai', 京东: 'JD.com', 阶跃星辰: 'StepFun', 遂原科技: 'Enflame', 燧原科技: 'Enflame',
  辉羲智能: 'Huixi Zhineng', 贝壳: 'Beike', 识渊科技: 'Shiyuan Keji', 虾皮: 'Shopee', 联想: 'Lenovo',
  网易: 'NetEase', 经纬恒润: 'HiRain Technologies', 米哈游: 'miHoYo', 格灵深瞳: 'DeepGlint', 智谱: 'Zhipu AI',
  易控智驾: 'Yikong Zhijia', 数坤科技: 'Shukun Technology', 摩尔线程: 'Moore Threads', 拼多多: 'Pinduoduo',
  小马智行: 'Pony.ai', 小光子: 'Xiaoguangzi', 寒武纪: 'Cambricon', 好未来: 'TAL Education', 商汤: 'SenseTime',
  后摩智能: 'Houmo AI', 原粒半导体: 'Yuanli Semiconductor', 南湖研究院: 'Nanhu Laboratory', 北极雄芯: 'Beiji Xiongxin',
  传音: 'Transsion', 中科类脑: 'Zhongke Leinao', 中科曙光: 'Sugon', 中兴: 'ZTE', 上海AI实验室: 'Shanghai AI Lab',
  B站: 'Bilibili', Teleai: 'TeleAI',
  小厂: { en: 'Smaller companies', vi: 'Công ty nhỏ' },
  综合: { en: 'General', vi: 'Tổng hợp' },
};

export function companyName(company: string, lang: Lang): string {
  if (lang === 'zh') return company;
  const name = COMPANY_NAMES[company];
  return (typeof name === 'string' ? name : name?.[lang]) ?? company;
}
