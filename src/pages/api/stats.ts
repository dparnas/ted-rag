import type { NextApiRequest, NextApiResponse } from "next";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  res.status(200).json({
    chunk_size: 512,
    overlap_ratio: 0.1,
    top_k: 7,
  });
}
