# Smart Invoice HCR - Vercel Deployment

## Quick Deploy to Vercel

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel:**
   ```bash
   vercel login
   ```

3. **Deploy:**
   ```bash
   cd /Users/iamdheeraj/Desktop/DiaTrac/dheeru-doc-intel
   vercel
   ```

4. **Set Environment Variables:**
   After first deployment, add your Azure secrets:
   ```bash
   vercel env add AZURE_FORM_ENDPOINT
   # Paste: https://dheeru-doc-intel.cognitiveservices.azure.com/
   
   vercel env add AZURE_FORM_KEY
   # Paste your Azure key
   
   vercel env add AZURE_MODEL_ID
   # Paste: handwritten_form_model_v2
   ```

5. **Redeploy with secrets:**
   ```bash
   vercel --prod
   ```

## Alternative: Deploy via Vercel Dashboard

1. Go to https://vercel.com/new
2. Import your GitHub repository: `dev-dheerajmaurya/smart-invoice-hcr`
3. Add Environment Variables in Settings:
   - `AZURE_FORM_ENDPOINT`
   - `AZURE_FORM_KEY`
   - `AZURE_MODEL_ID`
4. Deploy

## Important Notes

- **Storage limitation**: Vercel serverless functions are stateless. The `storage/` directory and SQLite database won't persist between requests.
- **For production**: Consider using:
  - AWS S3 / Azure Blob Storage for images
  - PostgreSQL / MongoDB for database
  - Or deploy to a VM/container service (Railway, Render, DigitalOcean)

## Limitations on Vercel

- ❌ No persistent filesystem (images/db lost after request)
- ❌ 50MB deployment size limit
- ❌ 10-second timeout for free tier
- ✅ Works for API testing
- ✅ Good for stateless operations

## Better Alternatives for This Project

**Recommended: Railway or Render**
- ✅ Persistent storage
- ✅ SQLite works
- ✅ Longer timeouts
- ✅ File uploads persist

Would you like me to set up deployment for Railway or Render instead?
